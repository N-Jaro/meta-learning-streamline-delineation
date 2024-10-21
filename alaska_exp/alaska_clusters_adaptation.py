import os
import numpy as np
import argparse
import tensorflow as tf
import datetime
import wandb
from sklearn.model_selection import train_test_split
from libs.loss import dice_loss

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
    print(csv_path)
    print(df['huc_code'])
    all_data = []
    all_labels = []
    
    for huc_code in df['huc_code']:
        print(os.path.join(data_dir, f'{huc_code}_data.npy'))
        print(os.path.join(data_dir, f'{huc_code}_label.npy'))
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

def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    parts = csv_filename.split('_')
    
    if "random" in parts:
        scenario_type = "random"
        num_clusters = parts[2]
        cluster_id = parts[4]
    else:
        scenario_type = "regular"
        num_clusters = parts[2]
        cluster_id = parts[4]

    scenario = {
        "type": scenario_type,
        "num_clusters": num_clusters,
        "cluster_id": cluster_id
    }
    return scenario

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

    return best_model

def create_save_directory(model_path, csv_file):
    """Creates a directory for saving the adapted model using the CSV filename."""
    # Extract the CSV filename without the extension
    csv_filename = os.path.splitext(os.path.basename(csv_file))[0]

    # Get the directory where the model is located
    model_dir = os.path.dirname(model_path)
    
    # Create the save directory based on the CSV filename
    save_dir = os.path.join(model_dir, csv_filename)
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir

def main(args):
    # Load and concatenate the data and labels
    combined_data, combined_labels = load_data_and_labels_from_csv(args.csv_path, args.data_dir, args.normalization_type)

    # Load the saved model
    model = load_model(args.model_path)

    # Create the directory for saving adapted model
    save_dir = create_save_directory(args.model_path, args.csv_path)

    # Extract scenario from CSV filename
    csv_filename = os.path.splitext(os.path.basename(args.csv_path))[0]
    scenario = extract_scenario_from_filename(csv_filename)

    # Generate run name combining scenario details and current date/time
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{scenario['type']}_{scenario['num_clusters']}clusters_cluster{scenario['cluster_id']}_{date_time}"

  # Adapt the model and save the best one, while logging to wandb with the run name
    best_adapted_model = adapt_model(model, combined_data, combined_labels, args.inner_steps, args.learning_rate, save_dir, args.wandb_project, run_name, scenario, args.patience)

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt MAML model to new data based on HUC codes in CSV.")
    parser.add_argument('--model_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/models/maml_3_500_1_20241002_114507/maml_model.keras', help='Path to the saved .keras model file')
    parser.add_argument('--csv_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/new_meta_train_huc_code/huc_code_kmean_5_cluster_0_adapt.csv', help='Path to the CSV file containing HUC codes')
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/', help='Directory containing the .npy files')
    parser.add_argument('--inner_steps', type=int, default=100, help='Number of adaptation steps')
    parser.add_argument('--learning_rate', type=float, default=0.00315, help='Learning rate for adaptation')
    parser.add_argument('--wandb_project', type=str, default="Alaska_maml_adaptation", help='WandB project name for logging')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help="Normalization type: '0', '-1', or 'none'")
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    args = parser.parse_args()

    main(args)