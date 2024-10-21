import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import wandb
from wandb.integration.keras import WandbCallback
import datetime
import glob

# Set random seeds for reproducibility
random.seed(44)
np.random.seed(44)
tf.random.set_seed(44)

# Define the channels to be used
channels = [0, 1, 2, 3, 4, 6, 7, 8]

def normalize_data(data, normalization_type='-1'):
    """Normalizes data based on the specified normalization type."""
    data_min = 0
    data_max = 255
    if normalization_type == '0':
        return (data - data_min) / (data_max - data_min)
    elif normalization_type == '-1':
        return 2 * ((data - data_min) / (data_max - data_min)) - 1
    elif normalization_type == 'none':
        return data
    else:
        raise ValueError("Unsupported normalization type. Choose '0', '-1', or 'none'.")

def load_data_and_labels_from_csv(csv_path, data_dir, normalization_type='-1'):
    """Loads data and labels from CSV and returns numpy arrays."""
    df = pd.read_csv(csv_path)
    all_data = []
    all_labels = []

    for huc_code in df['huc_code']:
        data_path = os.path.join(data_dir, f'{huc_code}_data.npy')
        label_path = os.path.join(data_dir, f'{huc_code}_label.npy')
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            print(f"Missing data or label file for HUC code: {huc_code}")
            continue  # Skip missing files

        # Load and process data
        data = np.load(data_path)[..., channels]
        data = normalize_data(data, normalization_type)
        labels = np.load(label_path)

        all_data.append(data)
        all_labels.append(labels)

    if not all_data or not all_labels:
        raise ValueError(f"No data found for CSV file: {csv_path}")

    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    return combined_data, combined_labels

def create_dataset(data, labels, batch_size, shuffle=True):
    """Creates a tf.data.Dataset from data and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def dice_loss(y_true, y_pred, smooth=1):
    """Dice loss function."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

class SimpleUNet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Downsample
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        # Bottleneck
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c1])
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.1)(c6)
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c6)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

def train_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args):
    """Trains the model and returns the trained model."""
    # Initialize WandB
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy'])

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    wandb_callback = WandbCallback(save_model=False)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, wandb_callback]
    )

    # Ensure the model save directory exists
    os.makedirs(args.model_save_dir, exist_ok=True)
    # Save the model
    model.save(args.model_save_path)
    print(f"Model saved at {args.model_save_path}")

    return model, history

def evaluate_model_per_huc(model, test_csv_path, data_dir, normalization_type, batch_size, metrics_save_path):
    """Evaluates the model and records metrics for each HUC code."""
    df = pd.read_csv(test_csv_path)
    huc_codes = df['huc_code']

    metrics_list = []

    for huc_code in huc_codes:
        data_path = os.path.join(data_dir, f'{huc_code}_data.npy')
        label_path = os.path.join(data_dir, f'{huc_code}_label.npy')
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            print(f"Missing data or label file for HUC code: {huc_code}")
            continue

        data = np.load(data_path)[..., channels]
        data = normalize_data(data, normalization_type)
        labels = np.load(label_path)

        predictions = model.predict(data, batch_size=batch_size)
        predictions_binary = (predictions > 0.5).astype(np.uint8)

        y_true = labels.flatten()
        y_pred = predictions_binary.flatten()

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        metrics_list.append({
            'huc_code': huc_code,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        print(f"HUC Code: {huc_code}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    if not metrics_list:
        print(f"No evaluation metrics to save for {test_csv_path}")
        return

    # Convert metrics_list to a DataFrame and save as CSV
    metrics_df = pd.DataFrame(metrics_list)
    # Ensure the metrics save directory exists
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Metrics saved at {metrics_save_path}")

    # Create a WandB Table
    table = wandb.Table(dataframe=metrics_df)

    # Log the table to WandB
    wandb.log({'evaluation_metrics': table})


    # Log overall metrics
    avg_precision = np.mean(metrics_df['precision'])
    avg_recall = np.mean(metrics_df['recall'])
    avg_f1 = np.mean(metrics_df['f1_score'])

    wandb.summary['avg_precision'] = avg_precision
    wandb.summary['avg_recall'] = avg_recall
    wandb.summary['avg_f1_score'] = avg_f1

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")

def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    basename = os.path.basename(csv_filename)
    parts = basename.replace('.csv', '').split('_')
    scenario = {}
    if "random" in parts:
        scenario_type = "random"
        idx_random = parts.index("random")
        num_clusters = parts[idx_random + 1]
        cluster_id = parts[idx_random + 3]
    else:
        scenario_type = "regular"
        idx_code = parts.index("code")
        num_clusters = parts[idx_code + 1]
        cluster_id = parts[idx_code + 3]
    scenario = {
        "type": scenario_type,
        "num_clusters": num_clusters,
        "cluster_id": cluster_id
    }
    return scenario

def process_csv_files_in_folder(args):
    """Processes all train and test CSV files in the input folder."""
    # Find all train CSV files
    train_csv_files = glob.glob(os.path.join(args.input_folder, '*_train.csv'))

    if not train_csv_files:
        print(f"No train CSV files found in {args.input_folder}")
        return

    for train_csv_path in train_csv_files:
        # Determine corresponding test CSV file
        test_csv_path = train_csv_path.replace('_train.csv', '_test.csv')
        if not os.path.exists(test_csv_path):
            print(f"Test CSV file {test_csv_path} not found for train CSV {train_csv_path}")
            continue

        # Update args with the current train and test CSV paths
        args.train_csv_path = train_csv_path
        args.test_csv_path = test_csv_path

        # Extract scenario from CSV filename
        scenario = extract_scenario_from_filename(args.train_csv_path)
        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{scenario['type']}_{scenario['num_clusters']}_clusters_{scenario['cluster_id']}_{date_time}"
        args.wandb_run_name = run_name  # Update the wandb run name

        # Create the model save path
        model_filename = f"{run_name}.keras"
        args.model_save_path = os.path.join(args.model_save_dir, model_filename)

        # Create the metrics save path
        metrics_filename = f"{run_name}_evaluation_metrics.csv"
        args.metrics_save_path = os.path.join(args.metrics_save_dir, metrics_filename)

        print(f"\nProcessing {train_csv_path} and {test_csv_path}")
        print(f"Run name: {run_name}")

        try:
            main(args)
        except Exception as e:
            print(f"An error occurred while processing {train_csv_path} and {test_csv_path}: {e}")

def main(args):
    # Load training data
    print("Loading training data...")
    train_data, train_labels = load_data_and_labels_from_csv(args.train_csv_path, args.data_dir, args.normalization_type)
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.3, random_state=44)

    # Create datasets
    batch_size = args.batch_size
    print("Creating datasets...")
    train_dataset = create_dataset(X_train, y_train, batch_size)
    val_dataset = create_dataset(X_val, y_val, batch_size, shuffle=False)

    # Build the model
    print("Building the model...")
    unet = SimpleUNet(input_shape=X_train.shape[1:], num_classes=1)
    model = unet.build_model()
    # model.summary()

    # Train the model
    print("Training the model...")
    epochs = args.epochs
    learning_rate = args.learning_rate
    patience = args.patience
    model, history = train_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args)

    # Evaluate the model
    print("Evaluating the model on test data...")
    evaluate_model_per_huc(model, args.test_csv_path, args.data_dir, args.normalization_type, batch_size, args.metrics_save_path)

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model from scratch using HUC codes in CSV.")
    parser.add_argument('--input_folder', type=str,
                        default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_clusters/',
                        help='Input folder containing train and test CSV files')
    parser.add_argument('--data_dir', type=str,
                        default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/',
                        help='Directory containing the .npy files')
    parser.add_argument('--model_save_dir', type=str,
                        default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/models/train_scratch/',
                        help='Directory to save the trained models')
    parser.add_argument('--metrics_save_dir', type=str,
                        default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/results/',
                        help='Directory to save the evaluation metrics')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'],
                        help="Normalization type: '0', '-1', or 'none'")
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--wandb_project', type=str, default='Alaska_train_scratch', help='WandB project name')

    args = parser.parse_args()
    process_csv_files_in_folder(args)