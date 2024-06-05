import os
import argparse
import tensorflow as tf
from libs.data_util import MetaDataLoader
from libs.loss import dice_loss  # Assuming you have this in your libs directory

def adapt_to_new_task(base_model, support_data, support_labels, inner_lr=0.001, inner_steps=1):
    model_copy = tf.keras.models.clone_model(base_model)
    model_copy.set_weights(base_model.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=inner_lr)
    for _ in range(inner_steps):
        with tf.GradientTape() as tape:
            predictions = model_copy(support_data)
            loss = dice_loss(support_labels, predictions)
        gradients = tape.gradient(loss, model_copy.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))
    return model_copy

def evaluate_adapted_model(model, query_data, query_labels):
    predictions = model(query_data)
    loss = dice_loss(query_labels, predictions)
    return loss.numpy(), predictions  # Return predictions for visualization if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt MAML model to a new task")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved MAML model')
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--testing_locations', nargs='+', default=['Covington'], help='Locations for meta-testing')
    parser.add_argument('--num_samples_per_location', type=int, default=25, help='Number of samples for the meta-testing set')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=400, help='Number of inner loop steps')
    args = parser.parse_args()

    # Load pre-trained MAML model
    maml_model = tf.keras.models.load_model(os.path.join(args.model_path,'maml_model.keras'))

    # Load new task data (using your MetaDataLoader)
    data_loader = MetaDataLoader(args.data_dir, args.normalization_type)
    test_episodes = data_loader.create_multi_episodes(1, args.num_samples_per_location, args.testing_locations)

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
    adapted_model_path = os.path.join(args.model_path, f'adapted_{args.testing_locations[0]}.keras')
    adapted_model.save(adapted_model_path) 
    print(f"Adapted model saved at: {adapted_model_path}")