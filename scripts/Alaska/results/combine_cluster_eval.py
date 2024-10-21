import pandas as pd
import glob
import os

def extract_cluster_number(filename):
    # Extract the cluster number based on the pattern after 'cluster' and before '_'
    basename = os.path.basename(filename)
    parts = basename.split('_')
    return int(parts[3])


# Define the base directory to start the search from
base_dir = './'

# Define your parameters
num_clusters = "10"  # Can be "5" or any other number
case = "random"    # Can be "regular" or "random"

# Create the search pattern using the parameters
paths = f"*{case}_{num_clusters}_clusters_*.csv"
print(f"Search pattern: {paths}")

# Use glob to find all matching files based on the dynamic pattern
file_paths = glob.glob(os.path.join(base_dir, paths))
print(f"Found file paths: {file_paths}")

# Initialize an empty list to store DataFrames
dfs = []

# Loop through the found file paths
for file_path in file_paths:
    try:
        # Extract the cluster number from the file name
        cluster_number = extract_cluster_number(file_path)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Add the cluster number as a new column
        df['cluster'] = cluster_number
        
        # Append the DataFrame to the list
        dfs.append(df)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Concatenate all DataFrames in the list
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(f"combined_{case}_{num_clusters}clusters.csv", index=False)
    print(f"Combined file saved as 'combined_{case}_{num_clusters}clusters.csv'.")
    
else:
    print("No valid DataFrames to concatenate.")
