import pandas as pd
import os

# Load the CSV file into a DataFrame
df = pd.read_csv('/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/merged_huc_code_clusters.csv')
print("CSV file loaded successfully.")
print(df.head())  # Print first few rows for verification

# List of scenarios and clusters to filter
scenarios = ['5_kmean'] #, '5_random', '10_kmean', '10_random'

# Parameters to set percentage of huc_code for train and adapt phases
train_percentage = 0.10
adapt_percentage = 0.05  

# Create an output directory to save the files
output_dir = './'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Function to process the scenario and randomly select percentages of each cluster for train, adapt, and test
def process_scenario(scenario, cluster_size):
    cluster_col = f'{cluster_size}_{scenario}_cluster'
    
    print(f"\nProcessing scenario: {scenario} with {cluster_size} clusters.")
    print(f"Cluster column: {cluster_col}")
    
    # Get unique clusters, excluding cluster 99 (if present)
    unique_clusters = df[df[cluster_col] != 99][cluster_col].unique()
    print(f"Unique clusters for {scenario} ({cluster_size} clusters): {unique_clusters}")
    
    # DataFrame to store the combined train data for this scenario
    train_data_combined = pd.DataFrame(columns=['huc_code', 'cluster'])
    
    # DataFrame to store the total status (train, adapt, test) for each huc_code
    total_status = df[['huc_code']].copy()
    total_status['status'] = 'unknown'

    # For each cluster, select train, adapt, and test data
    for cluster in unique_clusters:
        print(f"\nProcessing cluster: {cluster}")
        cluster_data = df[df[cluster_col] == cluster]
        print(f"Cluster {cluster} has {len(cluster_data)} samples.")
        
        # Randomly select the percentage of train and adapt data
        train_data = cluster_data.sample(frac=train_percentage, random_state=42)
        remaining_data = cluster_data.drop(train_data.index)
        adapt_data = remaining_data.sample(frac=adapt_percentage / (1 - train_percentage), random_state=42)
        test_data = remaining_data.drop(adapt_data.index)
        
        # Mark the status of each huc_code as 'train', 'adapt', or 'test'
        total_status.loc[train_data.index, 'status'] = 'train'
        total_status.loc[adapt_data.index, 'status'] = 'adapt'
        total_status.loc[test_data.index, 'status'] = 'test'
        
        # Add the selected train data to the combined train DataFrame
        train_data_for_combined = train_data[['huc_code']].copy()
        train_data_for_combined['cluster'] = cluster
        train_data_combined = pd.concat([train_data_combined, train_data_for_combined], ignore_index=True)
        
        # Create the output filename for the adaptation samples for this cluster
        output_adapt_filename = f'huc_code_{scenario}_{cluster_size}_cluster_{cluster}_adapt.csv'
        output_adapt_filepath = os.path.join(output_dir, output_adapt_filename)
        
        # Save the adaptation data (huc_code) to a new CSV file for each cluster
        adapt_data[['huc_code']].to_csv(output_adapt_filepath, index=False)
        print(f"Saved adaptation data to: {output_adapt_filename}")
        
        # Create the output filename for the test samples for this cluster
        output_test_filename = f'huc_code_{scenario}_{cluster_size}_cluster_{cluster}_test.csv'
        output_test_filepath = os.path.join(output_dir, output_test_filename)
        
        # Save the test data (huc_code) to a new CSV file for each cluster
        test_data[['huc_code']].to_csv(output_test_filepath, index=False)
        print(f"Saved test data to: {output_test_filename}")
    
    # Save the combined train data for all clusters
    combined_train_filename = f'huc_code_{scenario}_{cluster_size}_train.csv'
    combined_train_filepath = os.path.join(output_dir, combined_train_filename)
    train_data_combined.to_csv(combined_train_filepath, index=False)
    print(f"Saved combined train data to: {combined_train_filename}")
    
    # Save the total status of all huc_codes (train/adapt/test) for this scenario
    total_status_filename = f'{scenario}_{cluster_size}_total.csv'
    total_status_filepath = os.path.join(output_dir, total_status_filename)
    total_status.to_csv(total_status_filepath, index=False)
    print(f"Saved total status of huc_codes to: {total_status_filename}")

# Process each scenario for both 5 and 10 clusters
for scenario in scenarios:
    cluster_size = int(scenario.split('_')[0])  # Extract the cluster size from the scenario name
    process_scenario(scenario.split('_')[1], cluster_size)

print('All files generated successfully!')
