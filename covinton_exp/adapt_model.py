import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
from libs.data_util import MetaDataLoader, JointDataLoader, visualize_predictions
from libs.loss import dice_loss  # Assuming you have this in your libs directory
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score

# --- Meta-training ---

def adapt_to_new_task(base_model, support_data, support_labels, inner_lr=0.001, inner_steps=1):
    model_copy = tf.keras.models.clone_model(base_model)
    model_copy.set_weights(base_model.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=inner_lr)

    for step in range(inner_steps):
        with tf.GradientTape() as tape:
            predictions = model_copy(support_data)
            loss = dice_loss(support_labels, predictions)
        gradients = tape.gradient(loss, model_copy.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))
        print(f"Inner Step: {step+1}, Loss: {loss.numpy()}")
        wandb.log({"step": step, "inner_step_loss": loss})

    return model_copy

def evaluate_adapted_model(model, query_data, query_labels):
    predictions = model(query_data)
    loss = dice_loss(query_labels, predictions)
    return loss.numpy(), predictions  # Return predictions for visualization if needed


# --- Evaluation Function ---

def load_data(data_dir, locations, num_samples, mode='test', batch_size=32, normalization_type="-1"):
    data_loader = JointDataLoader(data_dir, locations, num_samples, mode=mode, batch_size=batch_size, normalization_type=normalization_type)
    if mode == 'train':
        train_dataset, vali_dataset = data_loader.load_data()
        return train_dataset, vali_dataset
    elif mode == 'test':
        test_dataset = data_loader.load_data()
        return test_dataset

def evaluate_scores(model, test_dataset, predict_path):
    y_true = []
    y_pred = []
    prediction_patches = []

    for test_data, test_labels in test_dataset:
        predictions = model.predict(test_data)
        y_true.append(test_labels.numpy().flatten())
        y_pred.append((predictions > 0.5).astype(np.int32).flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    f1_stream = f1_score(y_true, y_pred, labels=[1], average='micro')
    precision_stream = precision_score(y_true, y_pred, labels=[1], average='micro')
    recall_stream = recall_score(y_true, y_pred, labels=[1], average='micro')
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    return precision_stream, recall_stream, f1_stream, cohen_kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt MAML model to a new task")
    parser.add_argument('--model_path', type=str, default="/u/nathanj/meta-learning-streamline-delineation/scripts/models/maml_1_500_1_20240617_135103", help='Path to the saved MAML model')
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--testing_locations', nargs='+', default=['Covington'], help='Locations for meta-testing')
    parser.add_argument('--num_samples_per_location', type=int, default=25, help='Number of samples for the meta-testing set')
    parser.add_argument('--num_episodes', type=int, default=4, help='Number of samples for the meta-testing set')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=400, help='Number of inner loop steps')
    parser.add_argument('--predict_path', type=str, default='predictions', help='Path to save predictions')
    args = parser.parse_args()

    config={"model_path": args.model_path,
        "data_dir": args.data_dir,
        "normalization_type": args.normalization_type,
        "testing_locations": args.testing_locations,
        "num_samples_per_location": args.num_samples_per_location,
        "num_episodes": args.num_episodes,
        "inner_lr": args.inner_lr,
        "inner_steps": args.inner_steps,
    }

    # Start a new wandb run for adaptation or continue the previous run
    wandb.init(project="maml_adaptation", name=f"adapt_{args.testing_locations[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)

    # Load pre-trained MAML model
    maml_model = tf.keras.models.load_model(os.path.join(args.model_path,'maml_model.keras'))

    # Load new task data (using your MetaDataLoader)
    data_loader = MetaDataLoader(args.data_dir, args.normalization_type)
    test_episodes = data_loader.create_multi_episodes(args.num_episodes, args.num_samples_per_location, args.testing_locations)

    # Adapt the model
    # Example usage:
    for test_episode in test_episodes:
        adapted_model = adapt_to_new_task(maml_model, 
                                        test_episode["support_set_data"], 
                                        test_episode["support_set_labels"], 
                                        inner_lr=args.inner_lr,
                                        inner_steps=args.inner_steps)

        # Evaluate the adapted model
        test_loss, predictions = evaluate_adapted_model(adapted_model, 
                                            test_episode["query_set_data"], 
                                            test_episode["query_set_labels"])

    print(f"Test Loss after adaptation: {test_loss}")

    # Save the adapted model
    adapted_model_path = os.path.join(args.model_path, f'adapted_{args.testing_locations[0]}_.keras')
    adapted_model.save(adapted_model_path) 
    print(f"Adapted model saved at: {adapted_model_path}")
    
    # Visualize predictions and log to WandB
    test_dataset = load_data(args.data_dir, [args.testing_locations[0]], num_samples=10, mode='test')
    fig = visualize_predictions(adapted_model, test_dataset, num_samples=10)
    wandb.log({"predictions": wandb.Image(fig)})

     # Evaluate scores
    test_dataset = load_data(args.data_dir, [args.testing_locations[0]], num_samples=None, mode='test')
    precision_stream, recall_stream, f1_stream, cohen_kappa = evaluate_scores(adapted_model, test_dataset, args.predict_path)
    print(f"Evaluation scores for location {args.testing_locations[0]}")
    print(f"Precision: {precision_stream}, Recall: {recall_stream}, F1-Score: {f1_stream}, Cohen Kappa: {cohen_kappa}")
    wandb.log({"test_f1_score": f1_stream, "test_precision_stream": precision_stream, "test_recall_stream": recall_stream, "test_cohen_kappa": cohen_kappa})

    