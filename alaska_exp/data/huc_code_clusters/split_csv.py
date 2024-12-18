import pandas as pd
import os

# Load the combined dataset
file_path = '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_clusters/merged_huc_code_clusters.csv'  # Update this with your actual file path
df = pd.read_csv(file_path)

# Define parameters for processing
cluster_types = ['5_kmean', '5_random', '10_kmean', '10_random']
splits = ['train', 'test']
output_dir = './'  # Update this with your desired output directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to generate cluster-specific CSV files with the correct naming format
def generate_cluster_files(df, cluster_type, split_type, output_dir):
    cluster_col = f'{cluster_type}_cluster'
    split_col = f'{cluster_type}_split'

    # Filter out clusters with a value of 99
    df_filtered = df[df[cluster_col] != 99]

    # print(df_filtered.columns)

    # Determine the prefix for the file name based on the cluster type
    if 'random' in cluster_type:
        prefix = 'huc_code_random'
    else:
        prefix = 'huc_code'
    
    # Extract the number of clusters from the cluster type (e.g., '5' or '10')
    num_clusters = cluster_type.split('_')[0]

    # Filter the DataFrame for the specific cluster type and split type
    for cluster_number in df_filtered[cluster_col].unique():
        filtered_df = df_filtered[(df_filtered[cluster_col] == cluster_number) & (df_filtered[split_col] == split_type)]

        if not filtered_df.empty:
            # Create the file name based on the prefix, number of clusters, cluster number, and split type
            file_name = f'{prefix}_{num_clusters}_cluster_{cluster_number}_{split_type}.csv'
            file_path = os.path.join(output_dir, file_name)

            # Save the filtered DataFrame to CSV
            filtered_df[['huc_code']].to_csv(file_path, index=False)

# Generate files for each combination of cluster type, cluster number, and split type
for cluster_type in cluster_types:
    for split_type in splits:
        generate_cluster_files(df, cluster_type, split_type, output_dir)

print("Files generated successfully.")
