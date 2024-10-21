import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from libs.data_util import MetaDataLoader
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet
from libs.loss import dice_coefficient, dice_loss
# Removed import of adapt_model as it's not used in Reptile

# --- Meta-training with Reptile ---

def train_task_model(model, inputs, outputs, optimizer):
    # Inner-loop training for a single task
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = dice_loss(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy()

def reptile_model(base_model, episodes, meta_lr=0.001, inner_lr=0.001,
                  inner_steps=5, epochs=1000, patience=15, save_path='models'):
    """
    Implements the Reptile meta-learning algorithm.

    Args:
        base_model: The model to be meta-trained.
        episodes: List of episodes/tasks for meta-training.
        meta_lr: Meta learning rate (epsilon in Reptile).
        inner_lr: Learning rate for the inner loop.
        inner_steps: Number of inner loop optimization steps.
        epochs: Number of epochs for meta-training.
        patience: Early stopping patience.
        save_path: Directory to save the trained model.

    Returns:
        Meta-trained model and path where the model is saved.
    """

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'reptile_{inner_steps}_{epochs}_{date_time}'
    model_path = os.path.join(save_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f'Initialize the Training process: {name}')

    # Initialize WandB
    wandb.init(project="reptile_experiment",
               name=name,
               config={
                   "meta_lr": meta_lr,
                   "inner_lr": inner_lr,
                   "inner_steps": inner_steps,
                   "epochs": epochs,
                   "patience": patience,
                   "model_path": model_path,
                   "model_type": "unet"
               })

    # Initialize the optimizer for the inner loop
    inner_optimizer = tf.keras.optimizers.SGD(learning_rate=inner_lr)

    # Get initial weights
    initial_weights = base_model.get_weights()

    best_loss = float('inf')
    no_improvement_count = 0  # Counter to track the number of epochs without improvement

    for epoch in range(epochs):
        task_losses = []
        for episode_index, episode in enumerate(episodes):
            # Copy model for task-specific training
            model_copy = tf.keras.models.clone_model(base_model)
            model_copy.set_weights(initial_weights)

            # Inner loop: Task-specific training
            support_data = episode["support_set_data"]
            support_labels = episode["support_set_labels"]
            for _ in range(inner_steps):
                loss = train_task_model(model_copy, support_data, support_labels, inner_optimizer)

            # After inner-loop training, get the updated weights
            updated_weights = model_copy.get_weights()

            # Meta-update: Move the initial weights towards the updated weights
            # θ ← θ + ε (θ'_task - θ)
            epsilon = meta_lr
            initial_weights = [iw + epsilon * (uw - iw) for iw, uw in zip(initial_weights, updated_weights)]
            base_model.set_weights(initial_weights)

            # Evaluate on the query set for logging purposes
            query_data = episode["query_set_data"]
            query_labels = episode["query_set_labels"]
            val_predictions = base_model(query_data, training=False)
            val_loss = dice_loss(query_labels, val_predictions)
            task_losses.append(val_loss.numpy())

            wandb.log({
                "epoch": epoch,
                "episode": episode_index,
                "task_loss": loss,
                "val_loss": val_loss.numpy()
            })

        mean_loss = np.mean(task_losses)
        wandb.log({
            "epoch": epoch,
            "mean_val_loss": mean_loss
        })

        # Early stopping and model saving
        if mean_loss < best_loss:
            best_loss = mean_loss
            no_improvement_count = 0
            base_model.save(os.path.join(model_path, 'reptile_model.keras'))  # Save the best model
            print(f"Saved new best model with validation loss: {best_loss}")

        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} consecutive epochs, stopping training.")
                break  # Stop training if no improvement in 'patience' number of epochs

        print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

    print(f"Completed training for maximum {epochs} epochs.")
    wandb.finish()
    return base_model, model_path

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reptile for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_location', type=int, default=25, help='Number of samples per location')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--training_locations', nargs='+', default=['Rowancreek', 'Covington'], help='Locations for meta-training')
    parser.add_argument('--testing_locations', nargs='+', default=['Covington'], help='Locations for meta-testing')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes')

    # Reptile hyperparameters
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate (epsilon in Reptile)')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')

    args = parser.parse_args()

    # 1. Create data
    data_loader = MetaDataLoader(args.data_dir, args.normalization_type)
    meta_train_episodes = data_loader.create_multi_episodes(args.num_episodes, args.num_samples_per_location, args.training_locations)

    # 2. Model creation
    if args.model == "unet":
        model_creator = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
    elif args.model == "attentionUnet":
        model_creator = AttentionUnet(input_shape=(224, 224, 8), output_mask_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model_creator.build_model()
    model.summary()

    # 3. Meta-training with Reptile
    reptile_trained_model, model_path = reptile_model(
        base_model=model,
        episodes=meta_train_episodes,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        epochs=args.epochs,
        patience=args.patience,
        save_path=args.save_path
    )

    print("The best Reptile-trained model is saved at:", model_path)
