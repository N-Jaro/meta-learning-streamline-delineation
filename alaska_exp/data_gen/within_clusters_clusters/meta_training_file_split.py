import pandas as pd
import os
import random

# Load the CSV file into a DataFrame
df = pd.read_csv('/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/merged_huc_code_clusters.csv')
print("CSV file loaded successfully.")
print(df.head())  # Print first few rows for verification

# List of scenarios and clusters to filter
scenarios = ['5_kmean', '5_random', '10_kmean', '10_random']
clusters = [5, 10]  # Number of clusters for each scenario

# Create an output directory to save the files
output_dir = './'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Function to process the scenario, filter train, and randomly select 50% of each cluster
def process_scenario(scenario, cluster_size):
    cluster_col = f'{cluster_size}_{scenario}_cluster'
    split_col = f'{cluster_size}_{scenario}_split'
    
    print(f"\nProcessing scenario: {scenario} with {cluster_size} clusters.")
    print(f"Cluster column: {cluster_col}, Split column: {split_col}")
    
    # Filter 'train' data for this scenario, ignoring cluster 99
    train_data = df[(df[split_col] == 'train') & (df[cluster_col] != 99)]
    print(f"Filtered train data for {scenario} ({cluster_size} clusters) ignoring cluster 99:")
    print(train_data[['huc_code', cluster_col]].head())  # Show sample of filtered data
    
    # Get unique clusters, excluding cluster 99
    unique_clusters = train_data[cluster_col].unique()
    print(f"Unique clusters for {scenario} ({cluster_size} clusters): {unique_clusters}")
    
    # DataFrame to store combined train data for this scenario
    combined_train_data = pd.DataFrame(columns=['huc_code', 'cluster'])
    
    # For each cluster, select X% of the train data and output to a CSV file
    X = 0.10 # 1% = 0.01 | 5% = 0.05 | 10% = 0.10 |25% = 0.25 
    for cluster in unique_clusters:
        print(f"\nProcessing cluster: {cluster}")
        cluster_data = train_data[train_data[cluster_col] == cluster]
        print(f"Cluster {cluster} has {len(cluster_data)} train samples.")
        
        # Randomly select X% of the train data
        selected_data = cluster_data.sample(frac=X, random_state=42)
        print(f"Selected {len(selected_data)} samples for cluster {cluster}.")
        
        # Get the remaining X% of the train data (for adaptation)
        remaining_data = cluster_data.drop(selected_data.index)
        print(f"Remaining {len(remaining_data)} samples for adaptation in cluster {cluster}.")
        
        # Add the selected train data to the combined DataFrame, keeping track of the cluster
        selected_data_for_combined = selected_data[['huc_code']].copy()
        selected_data_for_combined['cluster'] = cluster
        combined_train_data = pd.concat([combined_train_data, selected_data_for_combined], ignore_index=True)
        
        # Create the output filename for the adaptation samples
        output_adapt_filename = f'huc_code_{scenario}_{cluster_size}_cluster_{cluster}_adapt.csv'
        output_adapt_filepath = os.path.join(output_dir, output_adapt_filename)
        
        # Save the remaining HUC codes for adaptation to a new CSV file
        remaining_data[['huc_code']].to_csv(output_adapt_filepath, index=False)
        print(f"Saved remaining adaptation data to: {output_adapt_filename}")
    
    # Save the combined train data for this scenario
    combined_train_filename = f'huc_code_{scenario}_{cluster_size}_train.csv'
    combined_train_filepath = os.path.join(output_dir, combined_train_filename)
    combined_train_data.to_csv(combined_train_filepath, index=False)
    print(f"Saved combined train data to: {combined_train_filename}")

# Process each scenario for both 5 and 10 clusters
for scenario in scenarios:
    cluster_size = int(scenario.split('_')[0])  # Extract the cluster size from the scenario name
    process_scenario(scenario.split('_')[1], cluster_size)

print('All files generated successfully!')
