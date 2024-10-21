import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
from libs.alaskaNKDataloader import AlaskaNKMetaDataset  # Update this import
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet
from libs.loss import dice_loss
import gc

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Functions ---

def train_task_model(model, inputs, outputs, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = dice_loss(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model, loss.numpy()

def initialize_wandb(name, config):
    wandb.init(project="Alaska_maml_experiment", name=name, config=config)

def setup_model(model_type, input_shape, num_classes):
    if model_type == "unet":
        model_creator = SimpleUNet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "attentionUnet":
        model_creator = AttentionUnet(input_shape=input_shape, output_mask_channels=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_creator.build_model()

def maml_training(base_model, episodes, config):
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'maml_{config["inner_steps"]}_{config["epochs"]}_{config["meta_batch_size"]}_{date_time}'
    model_path = os.path.join(config['save_path'], name)
    os.makedirs(model_path, exist_ok=True)
    print(f'Initialize the Training process: {name}')

    initialize_wandb(name, config)

    inner_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['initial_inner_lr'], config['decay_steps'], config['decay_rate'], staircase=True)
    meta_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['initial_meta_lr'], config['decay_steps'], config['decay_rate'], staircase=True)
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr_schedule)

    best_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(config['epochs']):
        task_losses = []
        for batch_index in range(config['meta_batch_size']):
            task_updates = []
            for episode_index, episode in enumerate(episodes):
                model_copy = tf.keras.models.clone_model(base_model)
                model_copy.set_weights(base_model.get_weights())

                inner_optimizer = tf.keras.optimizers.SGD(learning_rate=inner_lr_schedule(epoch * len(episodes) + episode_index))
                support_data, support_labels = episode["support_set_data"], episode["support_set_labels"]
                episode_losses = []
                for _ in range(config['inner_steps']):
                    model_copy, loss = train_task_model(model_copy, support_data, support_labels, inner_optimizer)
                    episode_losses.append(loss)

                query_data, query_labels = episode["query_set_data"], episode["query_set_labels"]
                # val_predictions = model_copy(query_data)
                # val_loss = dice_loss(query_labels, val_predictions)

                with tf.GradientTape() as meta_tape:
                    meta_tape.watch(model_copy.trainable_variables)
                    new_val_loss = dice_loss(query_labels, model_copy(query_data))
                gradients = meta_tape.gradient(new_val_loss, model_copy.trainable_variables)
                task_losses.append(new_val_loss.numpy())

                wandb.log({
                    "epoch": epoch,
                    "episode": episode_index,
                    "eps_loss": tf.reduce_mean(episode_losses),
                    # "eps_val_loss": val_loss.numpy(),
                    "new_val_loss": new_val_loss.numpy()
                })

                mapped_gradients = [tf.identity(grad) for grad in gradients]
                task_updates.append((mapped_gradients, new_val_loss))

            if task_updates:
                num_variables = len(base_model.trainable_variables)
                mean_gradients = []
                for i in range(num_variables):
                    grads = [update[0][i] for update in task_updates if update[0][i] is not None]
                    if grads:
                        mean_grad = tf.reduce_mean(tf.stack(grads), axis=0)
                        mean_gradients.append(mean_grad)
                    else:
                        mean_gradients.append(None)

                gradients_to_apply = [(grad, var) for grad, var in zip(mean_gradients, base_model.trainable_variables) if grad is not None]
                if gradients_to_apply:
                    meta_optimizer.apply_gradients(gradients_to_apply)

        mean_loss = tf.reduce_mean(task_losses)
        wandb.log({"epoch": epoch, "mean_val_loss": mean_loss})

        if mean_loss < best_loss:
            best_loss = mean_loss
            no_improvement_count = 0
            base_model.save(os.path.join(model_path, 'maml_model.keras'))
            print(f"Saved new best model with validation loss: {best_loss}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= config['patience']:
                print(f"No improvement for {config['patience']} consecutive epochs, stopping training.")
                break

        print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

    print(f"Completed training for maximum {config['epochs']} epochs.")
    wandb.finish()
    return base_model, model_path

def main():
    wandb.init()

    config = {
        "data_dir": wandb.config.data_dir,
        "training_csv": wandb.config.training_csv,
        "testing_csv": wandb.config.testing_csv,
        "num_watersheds_per_episode": wandb.config.num_watersheds_per_episode,
        "num_samples_per_location": wandb.config.num_samples_per_location,
        "normalization_type": wandb.config.normalization_type,
        "num_episodes": wandb.config.num_episodes,
        "initial_meta_lr": wandb.config.meta_lr,
        "initial_inner_lr": wandb.config.inner_lr,
        "decay_steps": wandb.config.decay_steps,
        "decay_rate": wandb.config.decay_rate,
        "meta_batch_size": wandb.config.meta_batch_size,
        "inner_steps": wandb.config.inner_steps,
        "epochs": wandb.config.epochs,
        "patience": wandb.config.patience,
        "save_path": wandb.config.save_path,
        "model_type": wandb.config.model_type
    }

    dataset = AlaskaNKMetaDataset(data_dir=config['data_dir'], csv_file=config['training_csv'], normalization_type=config['normalization_type'], channels=[0,1,2,3,4,6,7,8],  verbose= False)
    meta_train_episodes = dataset.create_multi_episodes(num_episodes=config['num_episodes'], N=config['num_watersheds_per_episode'], K=config['num_samples_per_location'])

    model = setup_model(config['model_type'], input_shape=(128, 128, 8), num_classes=1)
    model.summary()

    maml_model, model_path = maml_training(model, meta_train_episodes, config)

    # Clean up memory
    tf.keras.backend.clear_session()
    del maml_model
    gc.collect()

    print("The best MAML model saved at:", model_path)

# # --- Argument Parsing ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="MAML for medical image segmentation")
#     parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_128', help='Path to data directory')
#     parser.add_argument('--training_csv', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huvc_code_clusters/huc_code_train.csv', help='Path to training CSV file')
#     parser.add_argument('--testing_csv', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huvc_code_clusters/huc_code_test.csv', help='Path to testing CSV file')
#     parser.add_argument('--num_watersheds_per_episode', type=int, default=10, help='Number of watersheds per episode')
#     parser.add_argument('--num_samples_per_location', type=int, default=5, help='Number of samples per location')
#     parser.add_argument('--normalization_type', type=str, default='none', choices=['-1', '0', 'none'], help='Normalization range')
#     parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes')

#     parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
#     parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
#     parser.add_argument('--inner_lr', type=float, default=0.03, help='Inner loop learning rate')
#     parser.add_argument('--decay_steps', type=int, default=30, help='Learning rate decay steps')
#     parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
#     parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')
#     parser.add_argument('--inner_steps', type=int, default=1, help='Number of inner loop steps')
#     parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
#     parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
#     parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')

#     args = parser.parse_args()
#     main(args)
