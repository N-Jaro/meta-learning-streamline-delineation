import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
from libs.data_util import MetaDataLoader
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet
from libs.loss import dice_coefficient, dice_loss
from adapt_model import adapt_to_new_task, evaluate_adapted_model


# --- Meta-training ---

def train_task_model(model, inputs, outputs, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = dice_loss(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model, loss.numpy()


def maml_model(base_model, episodes, initial_meta_lr=0.001, initial_inner_lr=0.001, decay_steps=1000, decay_rate=0.96, meta_batch_size=1, inner_steps=1, epochs=500, patience=15, save_path='models'):

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'maml_{inner_steps}_{epochs}_{meta_batch_size}_{date_time}'
    model_path = os.path.join(save_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f'Initialize the Training process: {name}')

    config={"initial_meta_lr": initial_meta_lr,
            "initial_inner_lr": initial_inner_lr,
            "decay_steps": decay_steps,
            "decay_rate": decay_rate,
            "meta_batch_size": meta_batch_size,
            "inner_steps": inner_steps,
            "epochs": epochs,
            "patience": patience,
            "model_path": model_path,
            "model_type": "unet"
        }

    # Initialize WandB
    wandb.init(project="maml_experiment",
               name=name,
               config=config)

    # Define learning rate schedules
    inner_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_inner_lr, decay_steps, decay_rate, staircase=True)
    meta_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_meta_lr, decay_steps, decay_rate, staircase=True)
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr_schedule)

    best_loss = float('inf')
    no_improvement_count = 0  # Counter to track the number of epochs without improvement

    for epoch in range(epochs):
        task_losses = []
        for batch_index in range(meta_batch_size):
            task_updates = []
            for episode_index, episode in enumerate(episodes):
                # Copy model for task-specific training
                model_copy = tf.keras.models.clone_model(base_model)
                model_copy.set_weights(base_model.get_weights())

                inner_optimizer = tf.keras.optimizers.SGD(learning_rate=inner_lr_schedule(epoch * len(episodes) + episode_index))

                # Inner loop: Task-specific adjustments with dynamic learning rate
                support_data = episode["support_set_data"]
                support_labels = episode["support_set_labels"]
                episode_losses = []
                for _ in range(inner_steps):
                    model_copy, loss = train_task_model(model_copy, support_data, support_labels, inner_optimizer)
                    episode_losses.append(loss)

                # Evaluate the adapted model on the query set
                query_data = episode["query_set_data"]
                query_labels = episode["query_set_labels"]
                val_predictions = model_copy(query_data)
                val_loss = dice_loss(query_labels, val_predictions)
                task_losses.append(val_loss.numpy())

                wandb.log({
                    "epoch": epoch,
                    "episode": episode_index,
                    "eps_loss": tf.reduce_mean(episode_losses),
                    "eps_val_loss": val_loss.numpy()
                })

                # Compute gradients for meta-update using the base model's variables
                with tf.GradientTape() as meta_tape:
                    meta_tape.watch(model_copy.trainable_variables)
                    new_val_loss = dice_loss(query_labels, model_copy(query_data))
                gradients = meta_tape.gradient(new_val_loss, model_copy.trainable_variables)

                # Map gradients back to the base model's variables
                mapped_gradients = [tf.identity(grad) for grad in gradients]
                task_updates.append((mapped_gradients, new_val_loss))

            # Outer loop: Update the base model using aggregated gradients from all tasks
            if task_updates:
                num_variables = len(base_model.trainable_variables)
                mean_gradients = []
                for i in range(num_variables):
                    grads = [update[0][i] for update in task_updates if update[0][i] is not None]
                    if grads:
                        mean_grad = tf.reduce_mean(tf.stack(grads), axis=0)
                        mean_gradients.append(mean_grad)
                    else:
                        mean_gradients.append(None)  # Handle the case where all gradients for a variable are None

                gradients_to_apply = [(grad, var) for grad, var in zip(mean_gradients, base_model.trainable_variables) if grad is not None]
                if gradients_to_apply:
                    meta_optimizer.apply_gradients(gradients_to_apply)

        mean_loss = tf.reduce_mean(task_losses)
        wandb.log({
            "epoch": epoch,
            "mean_val_loss": mean_loss
        })

        # Early stopping and model saving
        if mean_loss < best_loss:
            best_loss = mean_loss
            no_improvement_count = 0
            base_model.save(model_path+'/maml_model.keras')  # Save the best model
            print(f"Saved new best model with validation loss: {best_loss}")

        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} consecutive epochs, stopping training.")
                break  # Stop training if no improvement in 'patience' number of epochs

        print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

    wandb.finish()
    return base_model, model_path, name, config


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAML for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_location', type=int, default=25, help='Number of samples per location')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--training_locations', nargs='+', default=['Alexander', 'Rowancreek'], help='Locations for meta-training')
    parser.add_argument('--testing_locations', nargs='+', default=['Covington'], help='Locations for meta-testing')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes')

    # MAML hyperparameters
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')
    parser.add_argument('--inner_steps', type=int, default=1, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')
    parser.add_argument('--target_location', type=str, help='Target location for evaluation')
    parser.add_argument('--predict_path', type=str, default='predictions', help='Path to save predictions')

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

    # 3. Meta-training
    maml_model, model_path, name, config = maml_model(model, meta_train_episodes, initial_meta_lr=args.meta_lr,
                                        initial_inner_lr=args.inner_lr, decay_steps=args.decay_steps,
                                        decay_rate=args.decay_rate, meta_batch_size=args.meta_batch_size,
                                        inner_steps=args.inner_steps, epochs=args.epochs, patience=args.patience,
                                        save_path=args.save_path)

    #add source model to config of the target model.
    config["source_model"] = name
    print("the best MAML model saved at:", model_path)
