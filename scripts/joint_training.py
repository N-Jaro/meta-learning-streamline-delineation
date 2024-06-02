import os
import argparse
import wandb
import datetime
import tensorflow as tf
from data_loader import DataLoader  # Ensure this is the correct path to your DataLoader class
from libs.unet import SimpleUNet
from libs.attentionUnet import AttentionUnet
from libs.loss import dice_loss

# --- Data Loading and Preprocessing ---

def load_data(data_dir, num_samples, mode='train'):
    data_loader = DataLoader(data_dir, num_samples, mode=mode)
    if mode == 'train':
        train_dataset, vali_dataset = data_loader.load_data()
        return train_dataset, vali_dataset
    elif mode == 'test':
        test_dataset = data_loader.load_data()
        return test_dataset

# --- Argument Parsing ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint training for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_location', type=int, default=100, help='Number of samples per location')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')

    args = parser.parse_args()

    # 1. Load data
    train_dataset, vali_dataset = load_data(args.data_dir, args.num_samples_per_location, mode='train')
    test_dataset = load_data(args.data_dir, args.num_samples_per_location, mode='test')

    # 2. Model creation
    if args.model == "unet":
        model_creator = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
    elif args.model == "attentionUnet":
        model_creator = AttentionUnet(input_shape=(224, 224, 8), output_mask_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model_creator.build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
                  loss=dice_loss,
                  metrics=["accuracy"])
    model.summary()

    # Initialize WandB
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'joint_training_{date_time}'
    model_path = os.path.join(args.save_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f'Initialize the Training process: {name}')

    wandb.init(project="joint_training_experiment",
               name=name,
               config={
                   "initial_lr": args.initial_lr,
                   "decay_steps": args.decay_steps,
                   "decay_rate": args.decay_rate,
                   "epochs": args.epochs,
                   "patience": args.patience,
                   "model_path": model_path,
                   "model_type": args.model
               })

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, 'best_model.keras'), save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=model_path),
        wandb.keras.WandbCallback()
    ]

    # 3. Training
    history = model.fit(train_dataset,
                        validation_data=vali_dataset,
                        epochs=args.epochs,
                        callbacks=callbacks)

    print("The best model saved at:", model_path)
    
    # 4. Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
