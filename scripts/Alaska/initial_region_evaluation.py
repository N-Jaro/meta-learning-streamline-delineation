import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(2, num_channels // 2 + 1, figsize=(20, 10))
    
    # Plot each channel of the 10th input patch
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

# Load the saved model (.keras file)
model = load_model('/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/models/maml_3_500_1_20240909_121616/maml_model.keras')

# Load the CSV file containing huc_code list, only read the 'huc_code' column
huc_codes = pd.read_csv("/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huvc_code_clusters/huc_code_test.csv", usecols=['huc_code'])

# Convert the 'huc_code' column to a list
huc_codes_list = huc_codes['huc_code'].tolist()

# Define the channels to be used
channels = [0, 1, 2, 3, 4, 6, 7, 8]

# Set normalization type (e.g., '0' for [0, 1] normalization, '-1' for [-1, 1], or 'none')
normalization_type = '-1'

# Initialize lists to store performance metrics for each huc_code
metrics = []

# Loop through each huc_code, load data and label, predict, evaluate performance, and save plots
for huc_code in huc_codes_list:
    # Construct the file paths for the data and label based on the huc_code
    data_path = f'/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/{huc_code}_data.npy'
    label_path = f'/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128/{huc_code}_label.npy'
    
    # Load the data and label for the current huc_code
    X_test = np.load(data_path)   # Test patches for this huc_code, shape (num_samples, 128, 128, total_channels)
    y_test = np.load(label_path)  # Corresponding labels for this huc_code, shape (num_samples, 128, 128)

    # Select only the specific channels
    X_test = X_test[:, :, :, channels]  # Now X_test shape will be (num_samples, 128, 128, 8)

    # Normalize the data
    X_test = normalize_data(X_test, normalization_type)

    # Predict the test data
    y_pred_probs = model.predict(X_test)  # Predicted probabilities, shape (num_samples, 128, 128)

    # Convert predicted probabilities into binary class labels (assuming a threshold of 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int)  # Shape (num_samples, 128, 128)

    # Select the 10th input patch, prediction, and label for plotting
    if X_test.shape[0] >= 10:
        X_test_10th_patch = X_test[9]   # 10th input patch
        y_pred_10th_patch = y_pred[9]   # 10th prediction
        y_test_10th_patch = y_test[9]   # 10th label

        # Save the plot of the 10th patch channels, prediction, and label
        plot_patch_channels(huc_code, X_test_10th_patch, y_pred_10th_patch, y_test_10th_patch, f'{huc_code}.png', channels)

    # Flatten the predictions and labels for pixel-level evaluation
    y_pred_flat = y_pred.flatten()  # Flatten to (num_samples * 128 * 128)
    y_test_flat = y_test.flatten()  # Flatten to (num_samples * 128 * 128)

    # Calculate precision, recall, and F1-score for class 1
    precision = precision_score(y_test_flat, y_pred_flat, pos_label=1)
    recall = recall_score(y_test_flat, y_pred_flat, pos_label=1)
    f1 = f1_score(y_test_flat, y_pred_flat, pos_label=1)

    # Store the results for this huc_code
    metrics.append({
        'huc_code': huc_code,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Convert the metrics to a DataFrame
metrics_df = pd.DataFrame(metrics)

# Save the metrics DataFrame to a CSV file
metrics_df.to_csv('./huc_code_metrics.csv', index=False)

print("Evaluation results and plots saved successfully.")
