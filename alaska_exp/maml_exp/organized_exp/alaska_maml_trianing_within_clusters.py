import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
from libs.alaskaNKDataloader import AlaskaNKMetaDataset  # Update this import
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet, DeeperUnet, DeeperUnet_dropout, SimpleAttentionUNet
from libs.loss import dice_loss

# --- Functions ---

def train_task_model(model, inputs, outputs, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = dice_loss(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model, loss.numpy()

def initialize_wandb(name, config):
    wandb.init(project=config['wandb_project_name'], name=name, config=config)

def setup_model(model_type, input_shape, num_classes):
    if model_type == "unet":
        model_creator = SimpleUNet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "deeperUnet":
        model_creator = DeeperUnet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "deeperUnetDropout":
        model_creator = DeeperUnet_dropout(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "simpleAttentionUnet":
        model_creator = SimpleAttentionUNet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "simpleAttentionUnet":
        model_creator = SimpleAttentionUNet(input_shape=input_shape, num_classes=num_classes)
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

    # Set initial learning rates
    inner_lr = config['inner_lr']
    meta_lr = config['meta_lr']

    # Define optimizers
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    best_loss = float('inf')
    no_improvement_count = 0

    # Patience and reduction factor for both inner_lr and meta_lr
    inner_lr_patience = config['patience']-5  # Number of epochs with no improvement for inner_lr
    inner_lr_reduction_factor = 0.96  # Factor to reduce inner_lr when no improvement

    meta_lr_patience = config['patience']  # Number of epochs with no improvement for meta_lr
    meta_lr_reduction_factor = 0.96  # Factor to reduce meta_lr when no improvement

    # Counters to track epochs without improvement for inner_lr and meta_lr
    inner_lr_no_improvement_count = 0
    meta_lr_no_improvement_count = 0

    for epoch in range(config['epochs']):
        task_losses = []
        for batch_index in range(config['meta_batch_size']):
            task_updates = []
            for episode_index, episode in enumerate(episodes):
                model_copy = tf.keras.models.clone_model(base_model)
                model_copy.set_weights(base_model.get_weights())

                # Get current inner learning rate manually
                inner_optimizer = tf.keras.optimizers.Adam(learning_rate=inner_lr)

                support_data, support_labels = episode["support_set_data"], episode["support_set_labels"]
                episode_losses = []
                for _ in range(config['inner_steps']):
                    model_copy, loss = train_task_model(model_copy, support_data, support_labels, inner_optimizer)
                    episode_losses.append(loss)

                query_data, query_labels = episode["query_set_data"], episode["query_set_labels"]
                val_predictions = model_copy(query_data)
                val_loss = dice_loss(query_labels, val_predictions)

                with tf.GradientTape() as meta_tape:
                    meta_tape.watch(model_copy.trainable_variables)
                    new_val_loss = dice_loss(query_labels, model_copy(query_data))
                gradients = meta_tape.gradient(new_val_loss, model_copy.trainable_variables)
                task_losses.append(new_val_loss.numpy())

                wandb.log({
                    "epoch": epoch,
                    "episode": episode_index,
                    "eps_loss": tf.reduce_mean(episode_losses),
                    "eps_val_loss": val_loss.numpy(),
                    "new_val_loss": new_val_loss.numpy(),
                    "inner_lr": inner_lr,  # Log the current inner learning rate for this episode
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
        wandb.log({
            "epoch": epoch,
            "mean_val_loss": mean_loss,
            "meta_lr": meta_optimizer.learning_rate.numpy()  # Log the current meta learning rate
        })

        # Check if the validation loss improved
        if mean_loss < best_loss:
            best_loss = mean_loss
            no_improvement_count = 0
            inner_lr_no_improvement_count = 0  # Reset inner_lr counter on improvement
            meta_lr_no_improvement_count = 0   # Reset meta_lr counter on improvement
            base_model.save(os.path.join(model_path, 'maml_model.keras'))
            print(f"Saved new best model with validation loss: {best_loss}")
        else:
            no_improvement_count += 1
            inner_lr_no_improvement_count += 1
            meta_lr_no_improvement_count += 1
            print(f"No improvement in epoch {epoch + 1}. No improvement count: {no_improvement_count}")

        # If no improvement for 'inner_lr_patience' epochs, reduce the inner learning rate
        if inner_lr_no_improvement_count >= inner_lr_patience:
            old_inner_lr = inner_lr
            inner_lr = old_inner_lr * inner_lr_reduction_factor
            print(f"Reducing inner learning rate from {old_inner_lr} to {inner_lr}")
            inner_lr_no_improvement_count = 0  # Reset the no improvement counter for inner_lr

        # If no improvement for 'meta_lr_patience' epochs, reduce the meta learning rate
        if meta_lr_no_improvement_count >= meta_lr_patience:
            old_meta_lr = meta_optimizer.learning_rate.numpy()
            new_meta_lr = old_meta_lr * meta_lr_reduction_factor
            meta_optimizer.learning_rate.assign(new_meta_lr)
            print(f"Reducing meta learning rate from {old_meta_lr} to {new_meta_lr}")
            meta_lr_no_improvement_count = 0  # Reset the no improvement counter for meta_lr

                # Early stopping condition
        if no_improvement_count >= config['patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs with best validation loss: {best_loss}")
            break

        tf.keras.backend.clear_session()
        print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

    print(f"Completed training for maximum {config['epochs']} epochs.")
    wandb.finish()
    return base_model, model_path

def main(args):
    config = vars(args)

    #The old data in data_gen has 9 channels. We skipped only Geomorephons (channels=[0,1,2,3,4,6,7,8])
    #The new data in data_gen_2 has 11 channels. We skip ORI and Geomorephons (channels=[0,1,2,4,6,7,8,9,10])
    dataset = AlaskaNKMetaDataset(data_dir=config['data_dir'], csv_file=config['training_csv'], normalization_type=config['normalization_type'], channels=config['channels'], verbose= False)
    meta_train_episodes = dataset.create_multi_episodes(num_episodes=config['num_episodes'], N=config['num_watersheds_per_episode'], K=config['num_samples_per_location'])
    episode_record = dataset.get_episode_record()
    print("Record of watersheds used in each episode:", episode_record)

    model = setup_model(config['model'], input_shape=(128, 128,  len(config['channels'])), num_classes=1)
    model.summary()

    maml_model, model_path = maml_training(model, meta_train_episodes, config)

    print("The best MAML model saved at:", model_path)

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAML for medical image segmentation")

    #old data_gen has 9 channels
    # parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/huc_code_data_znorm_128', help='Path to data directory')

    #new data_gen_2 has 11 channels
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_wo_254_255/huc_code_data_znorm_128/', help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/wo_254_255/model', help='Path to save trained models')
    parser.add_argument('--wandb_project_name', type=str, default='test_maml_exp_2', help='Path to save trained models')

    parser.add_argument('--training_csv', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/huc_code_kmean_5_train.csv', help='Path to training CSV file')
    parser.add_argument('--num_watersheds_per_episode', type=int, default=1, help='Number of watersheds per episode')
    parser.add_argument('--num_samples_per_location', type=int, default=15, help='Number of samples per location')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--num_episodes', type=int, default=25, help='Number of episodes')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='Channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')

    #simpleUnet 
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeperUnet', 'deeperUnetDropout', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--inner_lr', type=float, default=0.00180, help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.00090, help='Meta learning rate')

    #SimpleAttentionUnet
    # parser.add_argument('--model', type=str, default='simpleAttentionUnet', choices=['unet', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    # parser.add_argument('--inner_lr', type=float, default=0.000780, help='Inner loop learning rate')
    # parser.add_argument('--meta_lr', type=float, default=0.000359, help='Meta learning rate')

    parser.add_argument('--decay_steps', type=int, default=500, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')                                 
    parser.add_argument('--inner_steps', type=int, default=3, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()

    main(args)
