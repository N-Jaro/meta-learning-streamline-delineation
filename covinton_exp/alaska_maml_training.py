import os
import argparse
import wandb
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Alaska.libs.alaskaDataoader import AlaskaMetaDataset  # Assuming the dataset class is saved in this file
from libs.unet import SimpleUNet
from libs.attentionUnet import AttentionUnet
from libs.loss import dice_coefficient, dice_loss

# --- Meta-training ---

def train_task_model(model, inputs, outputs, optimizer):
    model.train()
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = dice_loss(outputs, predictions)
    loss.backward()
    optimizer.step()
    return model, loss.item()

def maml_model(base_model, episodes, initial_meta_lr=0.001, initial_inner_lr=0.001, decay_steps=1000, decay_rate=0.96, meta_batch_size=1, inner_steps=1, epochs=500, patience=15, save_path='models'):

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'maml_{inner_steps}_{epochs}_{meta_batch_size}_{date_time}'
    model_path = os.path.join(save_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f'Initialize the Training process: {name}')

    # Initialize WandB
    wandb.init(project="maml_experiment",
               name=name,
               config={
                   "initial_meta_lr": initial_meta_lr,
                   "initial_inner_lr": initial_inner_lr,
                   "decay_steps": decay_steps,
                   "decay_rate": decay_rate,
                   "meta_batch_size": meta_batch_size,
                   "inner_steps": inner_steps,
                   "epochs": epochs,
                   "patience": patience,
                   "model_path": model_path,
                   "model_type": "unet"
               })

    # Define learning rate schedules
    inner_lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optim.SGD(base_model.parameters(), lr=initial_inner_lr), gamma=decay_rate)
    meta_optimizer = optim.Adam(base_model.parameters(), lr=initial_meta_lr)

    best_loss = float('inf')
    no_improvement_count = 0  # Counter to track the number of epochs without improvement

    for epoch in range(epochs):
        task_losses = []
        for batch_index in range(meta_batch_size):
            task_updates = []
            for episode_index, episode in enumerate(episodes):
                # Copy model for task-specific training
                model_copy = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
                model_copy.load_state_dict(base_model.state_dict())

                inner_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr_schedule.get_last_lr()[0])

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
                model_copy.eval()
                with torch.no_grad():
                    val_predictions = model_copy(query_data)
                    val_loss = dice_loss(query_labels, val_predictions)

                # Compute gradients for meta-update using the base model's variables
                meta_optimizer.zero_grad()
                new_val_loss = dice_loss(query_labels, model_copy(query_data))
                new_val_loss.backward()
                meta_optimizer.step()

                task_losses.append(new_val_loss.item())

                wandb.log({
                    "epoch": epoch,
                    "episode": episode_index,
                    "eps_loss": np.mean(episode_losses),
                    "eps_val_loss": val_loss.item(),
                    "new_val_loss": new_val_loss.item()
                })

            # Outer loop: Update the base model using aggregated gradients from all tasks
            mean_loss = np.mean(task_losses)
            wandb.log({"epoch": epoch, "mean_val_loss": mean_loss})

            if mean_loss < best_loss:
                best_loss = mean_loss
                no_improvement_count = 0
                torch.save(base_model.state_dict(), os.path.join(model_path, 'maml_model.pth'))  # Save the best model
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
    parser = argparse.ArgumentParser(description="MAML for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_huc_code', type=int, default=25, help='Number of samples per huc_code')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--training_huc_codes', nargs='+', default=['Rowancreek', 'Covington'], help='HUC codes for meta-training')  # Replace with actual HUC codes
    parser.add_argument('--testing_huc_codes', nargs='+', default=['Covington'], help='HUC codes for meta-testing')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes')

    # MAML hyperparameters
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
    parser.add_argument('--decay_steps', type=int, default=150, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')
    parser.add_argument('--inner_steps', type=int, default=1, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')

    args = parser.parse_args()

    # 1. Create data
    dataset = AlaskaMetaDataset(data_dir=args.data_dir)
    meta_train_episodes = dataset.create_multi_episodes(args.num_episodes, args.num_samples_per_huc_code, args.training_huc_codes)

    # 2. Model creation
    if args.model == "unet":
        model_creator = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
    elif args.model == "attentionUnet":
        model_creator = AttentionUnet(input_shape=(224, 224, 8), output_mask_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    base_model = model_creator.build_model()
    base_model = base_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    base_model.train()

    # 3. Meta-training
    trained_model, model_path = maml_model(base_model, meta_train_episodes, initial_meta_lr=args.meta_lr,
                                           initial_inner_lr=args.inner_lr, decay_steps=args.decay_steps,
                                           decay_rate=args.decay_rate, meta_batch_size=args.meta_batch_size,
                                           inner_steps=args.inner_steps, epochs=args.epochs, patience=args.patience,
                                           save_path=args.save_path)

    print("The best MAML model saved at:", model_path)
