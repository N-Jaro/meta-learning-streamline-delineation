import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger
from libs.data_util import MetaDataLoader, JointDataLoader, visualize_predictions
from libs.loss import dice_loss, dice_coefficient  # Assuming you have this in your libs directory
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score

# --- Meta-training ---
def train_and_save_model(model, train_dataset, vali_dataset, model_path, epochs, patience, initial_lr, decay_steps, decay_rate, save_name, save_weights_only=True):
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=decay_rate, patience=patience, min_lr=1e-9, verbose=1, mode='min'),
        tf.keras.callbacks.EarlyStopping(patience=patience+5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, save_name), save_best_only=True, save_weights_only=save_weights_only),
        tf.keras.callbacks.TensorBoard(log_dir=model_path),
        WandbMetricsLogger()
    ]

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss=dice_loss,
                  metrics=[dice_coefficient, "accuracy"])

    # Training
    history = model.fit(train_dataset,
                        validation_data=vali_dataset,
                        epochs=epochs,
                        callbacks=callbacks)

    return model

def evaluate_adapted_model(model, query_data, query_labels):
    predictions = model(query_data)
    loss = dice_loss(query_labels, predictions)
    return loss.numpy(), predictions  # Return predictions for visualization if needed

def visualize_predictions(model, test_dataset, num_samples=10):
    # Get random samples from the test dataset
    test_data, test_labels = next(iter(test_dataset.unbatch().batch(num_samples)))
    predictions = model.predict(test_data)

    fig, axes = plt.subplots(num_samples, 10, figsize=(25, num_samples * 2.5))
    for i in range(num_samples):
        for j in range(8):
            axes[i, j].imshow(test_data[i, :, :, j], cmap='gray')
            axes[i, j].set_title(f'Input {j+1}')
        axes[i, 8].imshow(test_labels[i, :, :], cmap='gray')
        axes[i, 8].set_title('Ground Truth')
        axes[i, 9].imshow(predictions[i, :, :], cmap='gray')
        axes[i, 9].set_title('Prediction')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    return fig

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
    parser.add_argument('--model_path', type=str, default="/u/nathanj/meta-learning-streamline-delineation/scripts/models/maml_1_150_1_20240617_214428", help='Path to the saved MAML model')
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--testing_locations', nargs='+', default=['Covington'], help='Locations for meta-testing')
    parser.add_argument('--num_samples_per_location', type=int, default=None, help='Number of samples for the meta-testing set')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of samples for the meta-testing set')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=1000, help='Number of inner loop steps')
    parser.add_argument('--predict_path', type=str, default='predictions', help='Path to save predictions')
    args = parser.parse_args()

    config=args

    # Start a new wandb run for adaptation or continue the previous run
    run_name = f"adapt_{os.path.basename(args.model_path)}_{args.testing_locations[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="maml_adaptation", name=run_name, config=config)

  # Load pre-trained MAML model
    maml_model = tf.keras.models.load_model(os.path.join(args.model_path,'maml_model.keras'))

    # Load data for the current location
    train_dataset, vali_dataset = load_data(args.data_dir, args.testing_locations, num_samples=args.num_samples_per_location, mode='train')

    # Train and save the best weights from the current location
    adapted_model = train_and_save_model(maml_model, train_dataset, vali_dataset, args.model_path, args.epochs, args.patience, args.initial_lr, args.decay_steps, args.decay_rate, 'adapted_Covington_best_model.h5', save_weights_only=False)

    # Save the adapted model
    adapted_model_path = os.path.join(args.model_path, f'adapted_{args.testing_locations[0]}_.keras')
    adapted_model.save(adapted_model_path) 
    print(f"Adapted model saved at: {adapted_model_path}")
    
    # Visualize predictions and log to WandB
    test_dataset = load_data(args.data_dir, args.testing_locations, num_samples=10, mode='test')
    fig = visualize_predictions(adapted_model, test_dataset, num_samples=10)
    wandb.log({"predictions": wandb.Image(fig)})

     # Evaluate scores
    test_dataset = load_data(args.data_dir, args.testing_locations, num_samples=None, mode='test')
    precision_stream, recall_stream, f1_stream, cohen_kappa = evaluate_scores(adapted_model, test_dataset, args.predict_path)
    print(f"Evaluation scores for location {args.testing_locations[0]}")
    print(f"Precision: {precision_stream}, Recall: {recall_stream}, F1-Score: {f1_stream}, Cohen Kappa: {cohen_kappa}")
    wandb.log({"test_f1_score": f1_stream, "test_precision_stream": precision_stream, "test_recall_stream": recall_stream, "test_cohen_kappa": cohen_kappa})

    