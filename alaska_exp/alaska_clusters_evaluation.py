import tensorflow as tf
from tensorflow.keras.models import load_model
from libs.loss import dice_loss
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import datetime
import os
import wandb

# Function to normalize data based on the specified normalization type
def normalize_data(data, normalization_type='0'):
    """Normalizes data based on the specified normalization type."""
    if normalization_type == '0':
        data_min = 0
        data_max = 255
        return (data - data_min) / (data_max - data_min)
    elif normalization_type == '-1':
        data_min = 0
        data_max = 255
        return 2 * ((data - data_min) / (data_max - data_min)) - 1
    elif normalization_type == 'none':
        return data
    else:
        raise ValueError("Unsupported normalization type. Choose '0', '-1', or 'none'.")

# Function to plot the channels of an input patch, prediction, and label
def plot_patch_channels(huc_code, X_test_patch, y_pred_patch, y_test_patch, save_path, channels):
    num_channels = X_test_patch.shape[-1]
    
    # Create a figure for the plot
    fig, axes = plt.subplots(2, (num_channels // 2) + 1, figsize=(20, 10))
    
    # Plot each channel of the input patch
    for i in range(num_channels):
        ax = axes.flatten()[i]
        ax.imshow(X_test_patch[:, :, i], cmap='gray')
        ax.set_title(f'Channel {channels[i] + 1}')
        ax.axis('off')
    
    # Plot the prediction
    axes.flatten()[num_channels].imshow(y_pred_patch, cmap='gray')
    axes.flatten()[num_channels].set_title('Prediction')
    axes.flatten()[num_channels].axis('off')
    
    # Plot the label
    axes.flatten()[num_channels + 1].imshow(y_test_patch, cmap='gray')
    axes.flatten()[num_channels + 1].set_title('Label')
    axes.flatten()[num_channels + 1].axis('off')
    
    # Save the figure as a PNG file
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to extract scenario information from the CSV filename
def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    parts = csv_filename.split('_')
    
    if "random" in parts:
        scenario_type = "random"
        num_clusters = parts[3]  # "5" or "10" in your examples
        cluster_id = parts[5]    # "4", "3", "5", or "0" in your examples
    else:
        scenario_type = "regular"
        num_clusters = parts[2]  # "5" or "10" in your examples
        cluster_id = parts[4]    # "3", "2", or "0" in your examples
    
    scenario = {
        "type": scenario_type,
        "num_clusters": num_clusters,
        "cluster_id": cluster_id
    }

    return scenario

# Main function to execute the evaluation
def main(model_path, csv_path, normalization_type):
    # Extract scenario information from the CSV filename
    csv_filename = os.path.basename(csv_path)
    scenario = extract_scenario_from_filename(csv_filename)
    
    # Get the current date and time for the folder name
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate the directory name
    scenario_dir = f"{scenario['type']}_{scenario['num_clusters']}clusters_cluster{scenario['cluster_id']}_{date_time}"
    print(scenario_dir)
    
    # Initialize WandB with the scenario_dir as the run name
    wandb.init(
        project="streamline-delineation-evaluation",
        config={
            "model_path": model_path,
            "csv_path": csv_path,
            "normalization_type": normalization_type
        },
        name=scenario_dir  # Use scenario_dir as the run name
    )
    
    # Define the full directory path
    full_dir_path = os.path.join('./clusters_evaluations', scenario_dir)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    
    # Load the saved model (.keras file)
    model = load_model(model_path)

    # Load the CSV file containing huc_code list, only read the 'huc_code' column
    huc_codes = pd.read_csv(csv_path, usecols=['huc_code'])
    
    # Convert the 'huc_code' column to a list
    huc_codes_list = huc_codes['huc_code'].tolist()
    
    # Define the channels to be used
    channels = [0, 1, 2, 3, 4, 6, 7, 8]
    
    # Initialize lists to store performance metrics for each huc_code
    metrics = []
    
    # Create a WandB table to log metrics for each huc_code
    table = wandb.Table(columns=["huc_code", "precision", "recall", "f1_score"])
    
    # Loop through each huc_code, load data and label, predict, evaluate performance, and save plots
    for huc_code in huc_codes_list:
        # Construct the file paths for the data and label based on the huc_code
        data_path = f'/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/{huc_code}_data.npy'
        label_path = f'/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/{huc_code}_label.npy'
        
        # Load the data and label for the current huc_code
        X_test = np.load(data_path)   # Shape: (num_samples, 128, 128, total_channels)
        y_test = np.load(label_path)  # Shape: (num_samples, 128, 128)
        
        # Select only the specific channels
        X_test = X_test[:, :, :, channels]  # Shape: (num_samples, 128, 128, 8)
        
        # Normalize the data
        X_test = normalize_data(X_test, normalization_type)
        
        # Predict the test data
        y_pred_probs = model.predict(X_test)  # Shape: (num_samples, 128, 128)
        
        # Convert predicted probabilities into binary class labels (threshold of 0.5)
        y_pred = (y_pred_probs > 0.5).astype(int)  # Shape: (num_samples, 128, 128)
        
        # Select the 10th input patch, prediction, and label for plotting
        if X_test.shape[0] >= 10:
            X_test_10th_patch = X_test[9]   # 10th input patch
            y_pred_10th_patch = y_pred[9]   # 10th prediction
            y_test_10th_patch = y_test[9]   # 10th label
            
            # Save the plot of the 10th patch channels, prediction, and label
            plot_filename = os.path.join(full_dir_path, f'{huc_code}.png')
            plot_patch_channels(huc_code, X_test_10th_patch, y_pred_10th_patch, y_test_10th_patch, plot_filename, channels)
        
        # Flatten the predictions and labels for pixel-level evaluation
        y_pred_flat = y_pred.flatten()
        y_test_flat = y_test.flatten()
        
        # Calculate precision, recall, and F1-score for class 1
        precision = precision_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
        recall = recall_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
        f1 = f1_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
        
        # Add the metrics for this huc_code to the WandB table
        table.add_data(huc_code, precision, recall, f1)
        
        # Store the results for this huc_code
        metrics.append({
            'huc_code': huc_code,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    # Log the WandB table
    wandb.log({"huc_code_metrics": table})
    
    # Convert the metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Generate the output filename for the CSV file
    output_filename = f"{scenario['type']}_{scenario['num_clusters']}clusters_cluster{scenario['cluster_id']}.csv"
    output_filepath = os.path.join(full_dir_path, output_filename)
    
    # Save the metrics DataFrame to the generated CSV file
    metrics_df.to_csv(output_filepath, index=False)
    
    # Log the output CSV path to WandB
    wandb.log({"output_csv_path": output_filepath})
    
    print(f"Evaluation results and plots saved successfully in the directory '{full_dir_path}'.")
    
    # Finish the WandB run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model on test data and save metrics.')
    parser.add_argument('--model_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/models/maml_3_500_1_20240909_121616/huc_code_5_cluster_0_train/best_adapted_model_20240920_172501.keras', help='Path to the saved .keras model file.')
    parser.add_argument('--csv_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_clusters/huc_code_5_cluster_0_test.csv', help='Path to the CSV file containing huc_codes.')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help='Normalization type: "0", "-1", or "none".')

    args = parser.parse_args()

    main(args.model_path, args.csv_path, args.normalization_type)
