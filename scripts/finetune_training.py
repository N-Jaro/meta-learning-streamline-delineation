import os
import argparse
import wandb
import datetime
import tensorflow as tf
import numpy as np
from libs.data_util import JointDataLoader, visualize_predictions
from libs.unet import SimpleUNet
from libs.attentionUnet import AttentionUnet
from libs.loss import dice_loss, dice_coefficient
from wandb.integration.keras import WandbMetricsLogger

# --- Data Loading and Preprocessing ---

def load_data(data_dir, locations, num_samples, mode='train', batch_size=32, normalization_type="none"):
    data_loader = JointDataLoader(data_dir, locations, num_samples, mode=mode, batch_size=batch_size, normalization_type=normalization_type)
    if mode == 'train':
        train_dataset, vali_dataset = data_loader.load_data()
        return train_dataset, vali_dataset
    elif mode == 'test':
        test_dataset = data_loader.load_data()
        return test_dataset

def train_and_save_model(model, train_dataset, vali_dataset, model_path, epochs, patience, initial_lr, decay_steps, decay_rate, save_name, save_weights_only=True):
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, save_name), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=save_weights_only),
        tf.keras.callbacks.TensorBoard(log_dir=model_path),
        WandbMetricsLogger()
    ]

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss = dice_loss,
                  metrics=[dice_coefficient,'accuracy'])

    # Training
    history = model.fit(train_dataset,
                        validation_data=vali_dataset,
                        epochs=epochs,
                        callbacks=callbacks)

    return model

# --- Argument Parsing ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_location', type=int, default=100, help='Number of samples per location')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')
    parser.add_argument('--source_locations', nargs='+', default=['Rowancreek', 'Alexander'], help='Locations for initial training')
    parser.add_argument('--target_location', type=str, default='Covington', help='Location for fine-tuning')

    args = parser.parse_args()

    # Initialize WandB
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f'finetune_training_{date_time}'
    model_path = os.path.join(args.save_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f'Initialize the Training process: {name}')

    wandb.init(project="finetune_training_experiment",
               resume="allow",
               sync_tensorboard=True,
               name=name,
               config={
                   "initial_lr": args.initial_lr,
                   "decay_steps": args.decay_steps,
                   "decay_rate": args.decay_rate,
                   "epochs": args.epochs,
                   "patience": args.patience,
                   "model_path": model_path,
                   "model_type": args.model,
                   "num_sample_per_location": args.num_samples_per_location,
                   "source_locations": args.source_locations,
                   "target_location": args.target_location
               })

    run_id = wandb.run.id  # Save the run ID for later use
    run_id_path = os.path.join(model_path, 'wandb_run_id.txt')
    with open(run_id_path, 'w') as f:
        f.write(run_id)

    # Model creation
    if args.model == "unet":
        model_creator = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
    elif args.model == "attentionUnet":
        model_creator = AttentionUnet(input_shape=(224, 224, 8), output_mask_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model_creator.build_model()
    model.summary()

    # Training and fine-tuning on source locations sequentially
    for i, location in enumerate(args.source_locations):
        print(f"Training on source location: {location}...")
        
        # Load data for current source location
        train_dataset, vali_dataset = load_data(args.data_dir, [location], args.num_samples_per_location, mode='train')

        # Train and save the best weights
        save_name = f'best_{location}_weights.keras'
        model = train_and_save_model(model, train_dataset, vali_dataset, model_path, args.epochs, args.patience, args.initial_lr, args.decay_steps, args.decay_rate, save_name, save_weights_only=True)

        if i < len(args.source_locations) - 1:
            wandb.finish()
            # Initialize WandB for next fine-tuning phase
            wandb.init(project="finetune_training_experiment",
                       resume="allow",
                       sync_tensorboard=True,
                       id=run_id,
                       name=name)
            # Load the best weights for the next phase of training
            model.load_weights(os.path.join(model_path, save_name))

    # Final fine-tuning on the target location
    print(f"Fine-tuning on the target location: {args.target_location}...")
    
    # Load data for target location
    train_dataset, vali_dataset = load_data(args.data_dir, [args.target_location], args.num_samples_per_location, mode='train')
    
    # Load the best weights from the last source location training
    final_source = args.source_locations[-1]
    model.load_weights(os.path.join(model_path, f'best_{final_source}_weights.keras'))

    # Fine-tune and save the entire model
    model = train_and_save_model(model, train_dataset, vali_dataset, model_path, args.epochs, args.patience, args.initial_lr, args.decay_steps, args.decay_rate, 'best_target_model.h5', save_weights_only=False)

    # Visualize predictions and log to WandB
    test_dataset = load_data(args.data_dir, [args.target_location], num_samples=10, mode='test')
    fig = visualize_predictions(model, test_dataset, num_samples=10)
    wandb.log({"predictions": wandb.Image(fig)})

    # Finish the fine-tuning run
    wandb.finish()
