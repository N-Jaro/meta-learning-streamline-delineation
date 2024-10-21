import os
import numpy as np
import argparse
import tensorflow as tf
import datetime
import wandb
from sklearn.model_selection import train_test_split
from libs.loss import dice_loss
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the channels to be used
channels = [0, 1, 2, 3, 4, 6, 7, 8]

# Function to normalize data based on the specified normalization type
def normalize_data(data, normalization_type='-1'):
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

# --- Functions ---
def load_model(model_path):
    """Loads the saved Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})
    print(f"Loaded model from {model_path}")
    return model

def load_data_and_labels_from_csv(csv_path, data_dir, normalization_type):
    """Loads and concatenates the data and label .npy files listed in the CSV, selecting specific channels."""
    import pandas as pd  # Import pandas locally, as it's used in this function
    df = pd.read_csv(csv_path)
    all_data = []
    all_labels = []
    
    for huc_code in df['huc_code']:
        data_path = os.path.join(data_dir, f'{huc_code}_data.npy')
        label_path = os.path.join(data_dir, f'{huc_code}_label.npy')
        
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing data or label file for HUC code: {huc_code}")
        
        # Load the data and labels
        data = np.load(data_path)[..., channels]  # Select only the specified channels
        data = normalize_data(data, normalization_type)  # Normalize data
        labels = np.load(label_path)
        
        all_data.append(data)
        all_labels.append(labels)
        
        print(f"Loaded data and labels for HUC code: {huc_code}")

    # Concatenate all data and labels
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    print(f"Combined dataset shape: {combined_data.shape}, Combined labels shape: {combined_labels.shape}")
    
    return combined_data, combined_labels

# Function to extract scenario information from the CSV filename
def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    parts = csv_filename.split('_')
    
    if "random" in parts:
        scenario_type = "random"
        num_clusters = parts[3]  # "5" or "10" in your examples
        cluster_id = parts[5]    # "4", "3", "5", or "0" in your examples
    else:
        scenario_type = "kmean"
        num_clusters = parts[3]  # "5" or "10" in your examples
        cluster_id = parts[5]    # "3", "2", or "0" in your examples
    
    scenario = {
        "type": scenario_type,
        "num_clusters": num_clusters,
        "cluster_id": cluster_id
    }
    
    return scenario

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

def adapt_model(model, data, labels, inner_steps, learning_rate, save_path, project_name, run_name, scenario, patience):
    """Adapts the model to the input data and labels with training-validation split and saves the best model."""
    
    # Initialize wandb for logging with the specified run name
    wandb.init(project=project_name, config=scenario, name=run_name)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Split the dataset into training and validation sets (70/30 split)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=44)

    best_val_loss = float('inf')
    best_model = None
    no_improvement_count = 0  # Tracks how many steps without improvement

    # Adaptation loop
    for step in range(inner_steps):
        # Train on the training set
        with tf.GradientTape() as tape:
            predictions = model(train_data)
            train_loss = dice_loss(train_labels, predictions)
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Evaluate on the validation set
        val_predictions = model(val_data)
        val_loss = dice_loss(val_labels, val_predictions)

        # Log to wandb
        wandb.log({
            "step": step + 1,
            "train_loss": train_loss.numpy(),
            "val_loss": val_loss.numpy()
        })

        print(f"Step {step + 1}/{inner_steps}: Training Loss = {train_loss.numpy()}, Validation Loss = {val_loss.numpy()}")

        # Check if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
            no_improvement_count = 0  # Reset the counter if validation improves
            print(f"New best model found at step {step + 1}, Validation Loss: {best_val_loss.numpy()}")
        else:
            no_improvement_count += 1  # Increment the no improvement counter

        # Early stopping condition
        if no_improvement_count >= patience:
            print(f"Early stopping at step {step + 1} due to no improvement in validation loss for {patience} steps.")
            break

    # Save the best model
    if best_model is not None:
        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_save_path = os.path.join(save_path, 'best_adapted_model'+'_'+date_time+'.keras')
        os.makedirs(save_path, exist_ok=True)
        best_model.save(best_model_save_path)
        print(f"Best adapted model saved at {best_model_save_path}")

        # Log the model save path to WandB
        wandb.log({"best_model_save_path": best_model_save_path})

    else:
        print("No improvement in validation loss during training. No model saved.")

    # Finish wandb logging
    wandb.finish()

    return best_model, best_model_save_path

def evaluate_model(model, best_model_save_path, csv_path, data_dir, normalization_type, save_path, project_name, run_name, scenario):
    """Evaluates the model on the test set, saves plots, and logs results to WandB."""
    import pandas as pd  # Import pandas locally, as it's used in this function

    scenario["model_path"] = best_model_save_path
    scenario["csv_path"] = csv_path
    scenario["normalization_type"] = normalization_type
    scenario["save_path"] = save_path

    # Initialize WandB for logging with the specified run name
    wandb.init(project=project_name, config=scenario, name=f"{run_name}_evaluation")

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
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Loop through each huc_code, load data and label, predict, evaluate performance, and save plots
    for huc_code in huc_codes_list:
        # Construct the file paths for the data and label based on the huc_code
        data_path = os.path.join(data_dir, f'{huc_code}_data.npy')
        label_path = os.path.join(data_dir, f'{huc_code}_label.npy')
        
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
            plot_filename = os.path.join(save_path, f'{huc_code}.png')
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
        
        print(f"HUC code {huc_code}: Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
        
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
    output_filepath = os.path.join(save_path, output_filename)
    
    # Save the metrics DataFrame to the generated CSV file
    metrics_df.to_csv(output_filepath, index=False)
    
    # Log the output CSV path to WandB
    wandb.log({"output_csv_path": output_filepath})
    
    print(f"Evaluation results and plots saved successfully in the directory '{save_path}'.")
    
    # Finish the WandB run
    wandb.finish()

def process_adapt_and_evaluate(model_path, csv_folder, data_dir, inner_steps, learning_rate, wandb_project, normalization_type, patience, model_save_path):
    """Automates the adaptation and evaluation process for CSV files in a given folder."""
    
    # Check if the model_save_path exists
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"The specified model save path does not exist: {model_save_path}")

    # Get all CSV files in the folder
    csv_file_list = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

    for csv_file in csv_file_list:
        if csv_file.endswith('_adapt.csv'):
            print(f"Processing adaptation for {csv_file}")
            
            # Load and concatenate the data and labels
            combined_data, combined_labels = load_data_and_labels_from_csv(csv_file, data_dir, normalization_type)
            
            # Load the saved model
            model = load_model(model_path)
            
            # Create the directory for saving adapted model
            save_dir = create_save_directory(model_save_path, csv_file)

            # Extract scenario from CSV filename
            scenario = extract_scenario_from_filename(os.path.basename(csv_file))

            # Generate run name combining scenario details and current date/time
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{scenario['type']}_{scenario['num_clusters']}clusters_cluster{scenario['cluster_id']}_{date_time}"
            print(run_name)

            # Adapt the model
            best_adapted_model, best_model_save_path = adapt_model(model, combined_data, combined_labels, inner_steps, learning_rate, save_dir, "Alaska_maml_adaptation_within_clusters", run_name, scenario, patience)

            eval_save_dir =  os.path.join('./clusters_evaluations_within_clusters', run_name)
            # Find the corresponding test file
            test_file = csv_file.replace('_adapt.csv', '_test.csv')
            if test_file in csv_file_list:
                print(f"Evaluating adapted model on {test_file}")
                
                # Evaluate the adapted model on the test set
                evaluate_model(best_adapted_model, best_model_save_path, test_file, data_dir, normalization_type, eval_save_dir, "Alaska_maml_clusters_eval_within_clusters", run_name, scenario)
            else:
                print(f"Test file for {csv_file} not found.")

def create_save_directory(model_save_path, csv_file):
    """Creates a directory for saving the adapted model using the CSV filename and base model_save_path."""
    # Extract the CSV filename without the extension
    csv_filename = os.path.splitext(os.path.basename(csv_file))[0]

    # Create the save directory based on the CSV filename under model_save_path
    save_dir = os.path.join(model_save_path, csv_filename)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate adaptation and evaluation of MAML model based on HUC codes in CSV files.")
    parser.add_argument('--model_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/new_meta_train_huc_code/5_kmean_clusters/maml_model.keras', help='meta-learning model')
    parser.add_argument('--csv_folder', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/new_meta_train_huc_code/5_kmean_clusters/', help='Folder containing CSV files to process')
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/', help='Directory containing the .npy files')
    parser.add_argument('--inner_steps', type=int, default=200, help='Number of adaptation steps')
    parser.add_argument('--learning_rate', type=float, default=0.0035, help='Learning rate for adaptation')
    parser.add_argument('--wandb_project', type=str, default="Alaska_maml_adaptation_within_clusters", help='WandB project name for logging')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help="Normalization type: '0', '-1', or 'none'")
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--model_save_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/models/model_adapt_10022024/', help='Path to save adapted models')

    args = parser.parse_args()

    process_adapt_and_evaluate(args.model_path, args.csv_folder, args.data_dir, args.inner_steps, args.learning_rate, args.wandb_project, args.normalization_type, args.patience, args.model_save_path)
