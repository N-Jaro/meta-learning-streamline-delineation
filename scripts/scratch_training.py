import os
import argparse
import wandb
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score as sk_f1_score
from libs.data_util import JointDataLoader 
from libs.unet import SimpleUNet
from libs.attentionUnet import AttentionUnet
from libs.loss import dice_loss
from wandb.integration.keras import WandbMetricsLogger

# --- Data Loading and Preprocessing ---

def load_data(data_dir, locations, num_samples, mode='train', batch_size=32):
    data_loader = JointDataLoader(data_dir, locations, num_samples, mode=mode, batch_size=batch_size)
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
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, save_name), save_best_only=True, save_weights_only=save_weights_only),
        tf.keras.callbacks.TensorBoard(log_dir=model_path),
        WandbMetricsLogger()
    ]

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss=dice_loss,
                  metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)])

    # Training
    history = model.fit(train_dataset,
                        validation_data=vali_dataset,
                        epochs=epochs,
                        callbacks=callbacks)

    return model

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

def evaluate_f1_score(model, test_dataset):
    y_true = []
    y_pred = []
    for test_data, test_labels in test_dataset:
        predictions = model.predict(test_data)
        y_true.append(test_labels.numpy().flatten())
        y_pred.append((predictions > 0.5).astype(np.int32).flatten())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    f1 = sk_f1_score(y_true, y_pred, average='macro')

    return f1

# --- Argument Parsing ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline training for medical image segmentation from scratch")
    parser.add_argument('--data_dir', type=str, default='../samples/', help='Path to data directory')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--num_samples_per_location', type=int, help='Number of samples per location, if None use all samples')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models', help='Path to save trained models')
    parser.add_argument('--locations', nargs='+', default=['Rowancreek', 'Alexander', 'Covington'], help='Locations for training from scratch')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per location')

    args = parser.parse_args()

    for location in args.locations:
        for run in range(args.num_runs):
            # Initialize WandB
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f'scratch_training_{location}_run{run+1}_{date_time}'
            model_path = os.path.join(args.save_path, name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f'Initialize the Training process for location {location}, run {run+1}: {name}')

            wandb.init(project="scratch_training_experiment",
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
                           "location": location,
                           "run": run + 1
                       })

            # Load data for the current location
            train_dataset, vali_dataset = load_data(args.data_dir, [location], args.num_samples_per_location, mode='train')

            # Model creation
            if args.model == "unet":
                model_creator = SimpleUNet(input_shape=(224, 224, 8), num_classes=1)
            elif args.model == "attentionUnet":
                model_creator = AttentionUnet(input_shape=(224, 224, 8), output_mask_channels=1)
            else:
                raise ValueError(f"Unsupported model type: {args.model}")

            model = model_creator.build_model()
            model.summary()

            # Train and save the best weights from the current location
            model = train_and_save_model(model, train_dataset, vali_dataset, model_path, args.epochs, args.patience, args.initial_lr, args.decay_steps, args.decay_rate, f'best_{location}_model.h5', save_weights_only=False)

            # Visualize predictions and log to WandB
            test_dataset = load_data(args.data_dir, [location], num_samples=10, mode='test')
            fig = visualize_predictions(model, test_dataset, num_samples=10)
            wandb.log({"predictions": wandb.Image(fig)})

            # Evaluate F1-Score
            test_dataset = load_data(args.data_dir, [location], num_samples=None, mode='test')
            f1 = evaluate_f1_score(model, test_dataset)
            print(f"F1-Score for location {location}, run {run+1}: {f1}")
            wandb.log({"test_f1_score": f1})

            # Finish the training run
            wandb.finish()
