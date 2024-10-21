import pandas as pd

# Load the CSV files
df_5_clusters = pd.read_csv('./huc_code_5_clusters_min_size_10.csv')
df_10_clusters = pd.read_csv('./huc_code_10_clusters_min_size_10.csv')

# Rename columns in each dataframe
df_5_clusters.columns = ['huc_code', '5_kmean_cluster', '5_kmeans_split', '5_random_cluster', '5_random_split']
df_10_clusters.columns = ['huc_code', '10_kmean_cluster', '10_kmeans_split', '10_random_cluster', '10_random_split']

# Merge the dataframes on 'huc_code'
merged_df = pd.merge(df_5_clusters, df_10_clusters, on='huc_code', how='outer')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_huc_code_clusters.csv', index=False)

print("Merged file saved as 'merged_huc_code_clusters.csv'.")