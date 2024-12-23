import os
import glob
import wandb
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from libs.loss import dice_loss
import matplotlib.pyplot as plt
from libs.attentionUnet import AttentionUnet
from tensorflow.keras.models import load_model
from wandb.integration.keras import WandbCallback
from sklearn.model_selection import train_test_split
from libs.unet import SimpleUNet, DeeperUnet, DeeperUnet_dropout, SimpleAttentionUNet
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, jaccard_score, accuracy_score

# Set random seeds for reproducibility
# random.seed(44)
# np.random.seed(44)
# tf.random.set_seed(44)

# Function to plot the channels of an input patch, prediction, and label
def plot_patch_channels(huc_code, X_test_patch, y_pred_patch, y_pred_bin_patch, y_test_patch, save_path, channels):
    
    num_channels = len(channels)
    nrows = 2  # Fixed to 2 rows
    ncols = (num_channels + 3 + nrows - 1) // nrows  # Calculate the number of columns dynamically

    # Create a figure for the plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 8)) 
    
    # Plot each channel of the input patch
    for i in range(num_channels):
        ax = axes.flatten()[i]
        ax.imshow(X_test_patch[:, :, i], cmap='gray')
        # ax.set_title(f'Channel {channels[i] + 1}')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    # Plot the prediction
    axes.flatten()[num_channels].imshow(y_pred_patch, cmap='gray')
    axes.flatten()[num_channels].set_title('Prediction prob')
    axes.flatten()[num_channels].axis('off')
    
    axes.flatten()[num_channels + 1].imshow(y_pred_patch, cmap='gray')
    axes.flatten()[num_channels + 1].set_title('Prediction bin')
    axes.flatten()[num_channels + 1].axis('off')
    
    # Plot the label
    axes.flatten()[num_channels + 2].imshow(y_test_patch, cmap='gray')
    axes.flatten()[num_channels + 2].set_title('Label')
    axes.flatten()[num_channels + 2].axis('off')
    
    # Save the figure as a PNG file
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

def load_data_and_labels_from_csv(csv_path, data_dir,channels, samples_per_huc_code=-1, normalization_type='-1'):
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

        # Sample data (or use all if samples_per_huc_code is -1)
        if samples_per_huc_code == -1:
            sampled_data = data
            sampled_labels = labels
        else:
            indices = np.random.choice(data.shape[0], size=samples_per_huc_code, replace=False)
            sampled_data = data[indices]
            sampled_labels = labels[indices]

        all_data.append(sampled_data)
        all_labels.append(sampled_labels)

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

def train_init_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args, config):

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config, group=args.wandb_run_name, job_type="finetune_process")

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

    # Save the model
    model.save(args.init_model_file_path)
    print(f"Init model saved at {args.init_model_file_path}")

    wandb.finish()

    return model, history

def train_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args, config, cluster_id):

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

    # Save the model
    model.save(args.model_save_path)
    print(f"Adapted model saved at {args.model_save_path}")

    return model, history

def evaluate_model_per_huc(model, test_csv_path, data_dir, normalization_type, batch_size, metrics_save_path, channels, cluster_id):

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
        predictions_binary = (predictions > 0.33).astype(np.uint8)

        # Select the 10th input patch, prediction, and label for plotting
        indx = 20
        if data.shape[0] >= indx:
            X_test_patch = data[indx]   
            y_pred_patch = predictions[indx]   
            y_pred_bin_patch = predictions_binary[indx] 
            y_test_patch = labels[indx]   
            
            # Save the plot of the 10th patch channels, prediction, and label
            plot_filename = os.path.join(os.path.dirname(metrics_save_path), f'{huc_code}.png')
            plot_patch_channels(huc_code, X_test_patch, y_pred_patch, y_pred_bin_patch, y_test_patch, plot_filename, channels)

        y_true = labels.flatten()
        y_pred = predictions_binary.flatten()

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Calculate Intersection over Union (IoU) for class 1
        iou = jaccard_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)

        # Calculate Cohen's Kappa Coefficient
        kappa = cohen_kappa_score(y_true, y_pred)

        metrics_list.append({
            'huc_code': huc_code,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'IoU': iou,
            'kappa': kappa
        })

        print(f"HUC Code: {huc_code}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, IoU: {iou}, Kappa: {kappa}")

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
    wandb.log({f'evaluation_metrics_cluster_{cluster_id}': table})

    return metrics_df


def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    # huc_code_kmean_5_cluster_4_test.csv
    basename = os.path.basename(csv_filename)
    parts = basename.replace('.csv', '').split('_')
    scenario = {}
    if "random" in parts:
        scenario_type = "random"
        idx_random = parts.index("random")
        num_clusters = parts[idx_random + 1]
        cluster_id = parts[idx_random + 3]
    elif "kmean" in parts: 
        scenario_type = "kmean"
        idx_random = parts.index("kmean")
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

def prepare_env(args):
        
    # Ensure the model save directory exists
    os.makedirs(args.model_save_dir, exist_ok=True)  

    # Create tun_name for this run
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fine-tune_training_{args.num_sample_per_huc_code}_samples_{date_time}"
    args.wandb_run_name = run_name

    # create run save path: model_save_dir/{run_name}/
    args.run_model_save_dir = os.path.join(args.model_save_dir,run_name)
    os.makedirs(args.run_model_save_dir, exist_ok=True) 

    args.init_model_file_path=os.path.join(args.model_save_dir,run_name,"init_model.keras")
    print(args.init_model_file_path)

    config = vars(args) 

    return args, config

def main(args):

    args, config = prepare_env(args)

    # Get the training huc_code
    print("Getting training csv...")
    init_csv_file = glob.glob(os.path.join(args.input_folder, 'huc_code_*_train.csv'))
    print(f"Training csv file: {init_csv_file[0]}")

    # Load training data
    print("Loading training data...")
    init_data, init_labels = load_data_and_labels_from_csv(init_csv_file[0], args.data_dir, args.channels, args.num_sample_per_huc_code, args.normalization_type)
    print(f"Training data shape: {init_data.shape}")
    print(f"Training labels shape: {init_labels.shape}")

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(init_data, init_labels, test_size=0.3, random_state=44)

    # Create datasets
    batch_size = args.batch_size
    print("Creating datasets...")
    train_dataset = create_dataset(X_train, y_train, batch_size)
    val_dataset = create_dataset(X_val, y_val, batch_size, shuffle=False)

    # Build the model
    print("Building the model...")
    model = setup_model(config['model'], input_shape=(128, 128,  len(config['channels'])), num_classes=1)
    model.summary()

    # Train the model
    print("Training the initial model...")
    init_model, init_history = train_init_model(model, train_dataset, val_dataset, args.epochs, args.learning_rate, args.patience, args, config)

    """Trains the model and returns the trained model."""
    # Initialize WandB
    wandb.init(project=args.wandb_project, name=f"{args.wandb_run_name}_fine-tune", config=config, group=args.wandb_run_name)
    adapt_csv_files = glob.glob(os.path.join(args.input_folder, 'huc_code_*_adapt.csv'))
    all_test_metrics = []

    for adapt_csv_path in adapt_csv_files:

        # Determine corresponding test CSV file
        test_csv_path = adapt_csv_path.replace('_adapt.csv', '_test.csv')

        scenario = extract_scenario_from_filename(adapt_csv_path)
        args.metrics_save_path = os.path.join(args.run_model_save_dir,f"cluster_{scenario['cluster_id']}_eval.csv")
        args.model_save_path = os.path.join(args.run_model_save_dir,f"cluster_{scenario['cluster_id']}_model.keras")

        # Load training data
        print("Loading training data...")
        train_data, train_labels = load_data_and_labels_from_csv(adapt_csv_path, args.data_dir, args.channels, args.num_sample_per_huc_code, args.normalization_type)
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

        # Load init model
        init_model = load_model(args.init_model_file_path, custom_objects={"dice_loss": dice_loss})

        # Train the model
        print("Adapting the model...")
        trained_model, trained_history = train_model(init_model, train_dataset, val_dataset, args.epochs, args.learning_rate, args.patience, args, config, scenario['cluster_id'])

        # Evaluate the model
        print("Evaluating the model on test data...")
        metrics_df = evaluate_model_per_huc(trained_model, test_csv_path, args.data_dir, args.normalization_type, batch_size, args.metrics_save_path, args.channels, scenario['cluster_id'])

        if metrics_df is not None:  # Check if metrics_df is not None
            all_test_metrics.append(metrics_df)

    # Combine metrics from all test CSV files
    if all_test_metrics:
        combined_metrics_df = pd.concat(all_test_metrics, ignore_index=True)

        # Save the combined metrics DataFrame
        combined_metrics_save_path = os.path.join(args.run_model_save_dir, "combined_eval.csv")
        combined_metrics_df.to_csv(combined_metrics_save_path, index=False)
        print(f"Combined metrics saved at {combined_metrics_save_path}")

        # Create a WandB Table with all metrics
        table = wandb.Table(dataframe=combined_metrics_df)

        # Log the table to WandB only once
        wandb.log({'combined_evaluation_metrics': table}) 

        # Calculate average metrics
        avg_metrics = combined_metrics_df.mean(numeric_only=True)  # Calculate mean for numeric columns only

        # Log average metrics to WandB
        wandb.log({
            'avg_precision': avg_metrics['precision'],
            'avg_recall': avg_metrics['recall'],
            'avg_f1_score': avg_metrics['f1_score'],
            'avg_iou': avg_metrics['IoU'],
            'avg_kappa': avg_metrics['kappa']
        })

        # Print average metrics
        print(f"Average Precision: {avg_metrics['precision']}")
        print(f"Average Recall: {avg_metrics['recall']}")
        print(f"Average F1 Score: {avg_metrics['f1_score']}")
        print(f"Average IoU: {avg_metrics['IoU']}")
        print(f"Average Kappa: {avg_metrics['kappa']}")

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model from scratch using HUC codes in CSV.")
    parser.add_argument('--input_folder', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/within_clusters_clusters/5_kmean_clusters/', help='Input folder containing train and test CSV files')
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_wo_254_255/huc_code_data_znorm_128/', help='Directory containing the .npy files')
    parser.add_argument('--model_save_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/fine_tune_exp/wo_254_255/experiments/', help='Directory to save the trained models')
    parser.add_argument('--metrics_save_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/fine_tune_exp/wo_254_255/experiments/', help='Directory to save the evaluation metrics')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeperUnet', 'deeperUnetDropout', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    parser.add_argument('--num_sample_per_huc_code', type=int, default=-1, help='Number of samples per huc_code (-1 = all samples)')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help="Normalization type: '0', '-1', or 'none'")
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='Channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')

    parser.add_argument('--wandb_project', type=str, default='alaska_wo_254_255_fine-tune', help='WandB project name')

    args = parser.parse_args()
    main(args)

# Define the channels to be used
# channels = [0, 1, 2, 3, 4, 6, 7, 8]

#new data 
# channels = [0, 1, 2, 4, 6, 7, 8, 9, 10]