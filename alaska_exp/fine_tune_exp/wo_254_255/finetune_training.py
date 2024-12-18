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
from wandb.integration.keras import WandbCallback
from sklearn.model_selection import train_test_split
from libs.unet import SimpleUNet, DeeperUnet, DeeperUnet_dropout, SimpleAttentionUNet
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, jaccard_score, accuracy_score

# Set random seeds for reproducibility
random.seed(44)
np.random.seed(44)
tf.random.set_seed(44)

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

def setup_init_model(model_type, input_shape, num_classes):
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

def load_data_and_labels_from_csv(csv_path, data_dir,channels, normalization_type='-1'):
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

def train_init_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args, config):
    """Trains the model and returns the trained model."""
    # Initialize WandB
    # Create run name
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    init_run_name = f"init_model_{date_time}"
    args.init_run_name = init_run_name
    config['init_run_name'] = init_run_name

    #make sure that the save directory extists
    os.makedirs(os.path.join(args.model_save_dir, init_run_name), exist_ok=True)

    # create unique save path for each run and update in config and args
    init_model_file_path = os.path.join(args.model_save_dir, init_run_name, "init_model.keras")
    args.init_model_file_path = init_model_file_path
    config['init_model_file_path'] = init_model_file_path

    print(f"Init model will be saved at {init_model_file_path}")
    wandb.init(project=args.wandb_project, name=init_run_name, config=config)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=['Prscision', 'Recall', 'F1Score'])

    # Define callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint( filepath=init_model_file_path, monitor='val_loss', save_best_only=True, save_weights_only=False    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    wandb_callback = WandbCallback(save_model=False)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping, wandb_callback]
    )

    print(f"Model saved at {args.init_model_file_path}")

    return model, history, args.init_model_file_path

def adapt_model(init_model_path, data, label, learning_rate, patience, save_path, project_name, run_name, args, cofig){

    # Initialize wandb for logging with the specified run name
    wandb.init(project=project_name, config=scenario, name=run_name)

    # Split the dataset into training and validation sets (70/30 split)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=44)

    # Load the model
    print(f"Loading model from {init_model_path}...")
    model = tf.keras.models.load_model(init_model_path)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=['Prscision', 'Recall', 'F1Score'])


    # Define callbacks
    adated_model_save_path = os.path.join(save_path, "adated_model.keras")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint( filepath=adated_model_save_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    wandb_callback = WandbCallback(save_model=False)

    # Train the model
    print("Starting fine-tuning...")
    history = model.fit( train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[model_checkpoint, early_stopping, wandb_callback] )
    
    return best_model, model_save_path
}

def finetune_eval_clusters(init_model_path, cluster_csv_path, data_dir, args, config) {
    """Processing culusters data and adapt the initial model to the clusters"""
    # Find all adapt CSV files
    adapt_csv_files = glob.glob(os.path.join(args.eval_huc_code_dir, '*_adapt.csv'))
    
    if not adapt_csv_files:
        print(f"No adapt CSV files found in {args.input_folder}")
        return

    for adapt_csv_path in adapt_csv_files:

        # Determine corresponding test CSV file
        test_csv_path = adapt_csv_path.replace('_adapt.csv', '_test.csv')
        if not os.path.exists(test_csv_path):
            print(f"Test CSV file {test_csv_path} not found for train CSV {adapt_csv_path}")
            continue
        
        # create run name for fine-tuning phase
        cluster_id = adapt_csv_path.split('_')[5]
        print(f"Processing cluster: {cluster}")
        run_prefix = args.init_run_name.replace("init_","")
        ft_run_name = f"{run_prefix}_cluster_{cluster_id}
        eval_save_dir = os.path.joing(os.path.dirname(args.init_model_file_path),ft_run_name)

        # Load and concatenate the data and labels
        adapt_data, adapt_labels = load_data_and_labels_from_csv(adapt_csv_path, data_dir, args.channels, args.normalization_type)

        # adapt the init model to cluster
        best_model, best_model_save_path = adapt_model(init_model_path, adapt_data, adapt_labels, args.learning_rate, args.patience, eval_save_dir, args.project_name, ft_run_name, args, config)


}

def evaluate_model_per_huc(model, test_csv_path, data_dir, normalization_type, batch_size, metrics_save_path, channels):
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
        if data.shape[0] >= 20:
            X_test_patch = data[20]   
            y_pred_patch = predictions[20]   
            y_pred_bin_patch = predictions_binary[20] 
            y_test_patch = labels[20]   
            
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
    wandb.log({'evaluation_metrics': table})

    # Log overall metrics
    avg_precision = np.mean(metrics_df['precision'])
    avg_recall = np.mean(metrics_df['recall'])
    avg_f1 = np.mean(metrics_df['f1_score'])
    avg_iou = np.mean(metrics_df['IoU'])
    avg_kappa = np.mean(metrics_df['kappa'])

    wandb.summary['avg_precision'] = avg_precision
    wandb.summary['avg_recall'] = avg_recall
    wandb.summary['avg_f1_score'] = avg_f1
    wandb.summary['avg_iou'] = avg_iou
    wandb.summary['avg_kappa'] = avg_kappa

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}, Average IoU: {avg_iou}, Average Kappa: {avg_kappa}")

def main(args):

    config = {
        "data_dir": args.data_dir,
        "init_huc_code_csv": args.init_huc_code_csv,
        "eval_huc_code_dir": args.eval_huc_code_dir,
        "model_save_dir": args.model_save_dir,
        "metrics_save_dir": args.metrics_save_dir,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "normalization_type": args.normalization_type,
        "channels": args.channels,
        "wandb_project": args.wandb_project
    }


    # 1) train init model with train.csv
    # Load training data
    print("Loading training data...")
    train_data, train_labels = load_data_and_labels_from_csv(args.init_huc_code_csv, args.data_dir, args.channels, args.normalization_type)
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
    model = setup_init_model(config['model'], input_shape=(128, 128,  len(config['channels'])), num_classes=1)
    model.summary()

    # Train the model
    print("Training the model...")
    epochs = args.epochs
    learning_rate = args.learning_rate
    patience = args.patience
    model, history, init_model_path = train_init_model(model, train_dataset, val_dataset, epochs, learning_rate, patience, args, config)

    #2) Fine-tune to clusters

    ft_model,history = finetune_eval_clusters(init_model_path, args.eval_huc_code_dir, args.data_dir, args, cofig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model from scratch using HUC codes in CSV.")
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_wo_254_255/huc_code_data_znorm_128/', help='Directory containing the .npy files')

    parser.add_argument('--init_huc_code_csv', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/huc_code_kmean_5_train.csv', help='path to _train.csv')
    parser.add_argument('--eval_huc_code_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/', help='contain adapt and test csv')

    parser.add_argument('--model_save_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/fine_tune_exp/wo_254_255/model/', help='Directory to save the trained models')
    parser.add_argument('--metrics_save_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/fine_tune_exp/wo_254_255/model/', help='Directory to save the evaluation metrics')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeperUnet', 'deeperUnetDropout', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help="Normalization type: '0', '-1', or 'none'")
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='Channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')

    parser.add_argument('--wandb_project', type=str, default='alaska_wo_254_255_ft', help='WandB project name')

    args = parser.parse_args()
    

# Define the channels to be used
# channels = [0, 1, 2, 3, 4, 6, 7, 8]

#new data 
# channels = [0, 1, 2, 4, 6, 7, 8, 9, 10]