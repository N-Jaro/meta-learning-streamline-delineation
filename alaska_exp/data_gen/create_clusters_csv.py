import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['huc_code'] = df['huc_code'].astype(int)  # Ensure huc_code is integer
    return df

# Separate excluded HUC codes
def separate_excluded_hucs(df, exclude_huc_codes):
    excluded_df = df[df['huc_code'].isin(exclude_huc_codes)]
    df = df[~df['huc_code'].isin(exclude_huc_codes)]
    return df, excluded_df

# Apply K-Means clustering
def apply_kmeans_clustering(df, n_clusters):
    X = df.drop('huc_code', axis=1)
    X.fillna(X.mean(), inplace=True)  # Handle missing values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(X)
    return df, X, kmeans

# Reassign small clusters
def reassign_small_clusters(df, kmeans, min_cluster_size):
    cluster_sizes = df['cluster'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
    large_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index

    for cluster in small_clusters:
        small_cluster_points = df[df['cluster'] == cluster]
        for idx, point in small_cluster_points.iterrows():
            distances = kmeans.transform([point.drop(['huc_code', 'cluster'])])
            nearest_large_cluster = large_clusters[np.argmin(distances[0][large_clusters])]
            df.at[idx, 'cluster'] = nearest_large_cluster

    return df

# Reorganize cluster labels to be sequential (0, 1, 2, ...)
def reorganize_clusters(df):
    unique_clusters = sorted(df['cluster'].unique())
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    df['cluster'] = df['cluster'].map(cluster_mapping)
    return df, len(unique_clusters)

# Create train/test split
def create_train_test_split(df, cluster_col, split_col, train_ratio=0.10):
    df[split_col] = 'test'  # Default to 'test'
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster]
        train_size = int(len(cluster_data) * train_ratio)
        train_indices = cluster_data.sample(train_size, random_state=0).index
        df.loc[train_indices, split_col] = 'train'
    return df

# Create random clusters
def create_random_clusters(df, final_n_clusters):
    random_cluster_sizes = df['cluster'].value_counts()
    df['random_cluster'] = np.nan
    remaining_indices = df.index.tolist()

    for i in range(final_n_clusters):
        cluster_size = random_cluster_sizes.iloc[i]
        selected_indices = np.random.choice(remaining_indices, size=cluster_size, replace=False)
        df.loc[selected_indices, 'random_cluster'] = i
        remaining_indices = list(set(remaining_indices) - set(selected_indices))

    df['random_cluster'] = df['random_cluster'].astype(int)
    return df

# Add excluded HUCs back to the output
def add_excluded_hucs_back(output_df, excluded_df):
    excluded_df['kmean_cluster'] = 99
    excluded_df['random_cluster'] = 99
    excluded_df['kmeans_split'] = 'test'
    excluded_df['random_split'] = 'test'
    final_output_df = pd.concat([output_df, excluded_df[['huc_code', 'kmean_cluster', 'kmeans_split', 'random_cluster', 'random_split']]], ignore_index=True)
    return final_output_df

# Plot clusters using PCA or t-SNE
def plot_clusters(df, X, cluster_col, title, file_name):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df['pca_one'] = pca_result[:, 0]
    df['pca_two'] = pca_result[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='pca_one', y='pca_two',
        hue=cluster_col,
        palette=sns.color_palette('hsv', df[cluster_col].nunique()),
        data=df[df[cluster_col].notnull()],
        legend='full',
        alpha=0.8
    )
    plt.title(f'PCA of {title} Clusters')
    plt.savefig(file_name)
    plt.close()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(X)
    df['tsne_one'] = tsne_result[:, 0]
    df['tsne_two'] = tsne_result[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tsne_one', y='tsne_two',
        hue=cluster_col,
        palette=sns.color_palette('hsv', df[cluster_col].nunique()),
        data=df[df[cluster_col].notnull()],
        legend='full',
        alpha=0.8
    )
    plt.title(f't-SNE of {title} Clusters')
    plt.savefig(file_name.replace('pca', 'tsne'))
    plt.close()

# Main execution function
def main():
    # Parameters
    file_path = '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/average_values_cleaned.csv'
    exclude_huc_codes = [
        190503021001, 190503021002, 190503021003, 190503021004, 190503021005,
        190503021006, 190503021007, 190503021008, 190503021009, 190503021101,
        190503021102, 190503021103, 190503021104, 190503021105, 190503021106,
        190503021107, 190503021108, 190503021109, 190503021201, 190503021202,
        190503021203, 190503021204, 190503021205, 190503021206, 190503021207,
        190503021208, 190503021209, 190503021210, 190503021211, 190503021212,
        190503021300, 190503021401, 190503021402, 190503021403, 190503021404,
        190503021405, 190503021406, 190503021407, 190503021408, 190503021501,
        190503021502, 190503021503, 190503021504, 190503021505, 190503021506,
        190503021507, 190503021508, 190503021509, 190503021510, 190503021511
    ]
    n_clusters = 13
    min_cluster_size = 10

    # Load and process data
    df = load_data(file_path)
    df, excluded_df = separate_excluded_hucs(df, exclude_huc_codes)

    # Apply KMeans clustering
    df, X, kmeans = apply_kmeans_clustering(df, n_clusters)

    # Reassign small clusters
    df = reassign_small_clusters(df, kmeans, min_cluster_size)

    # Reorganize clusters to be sequential
    df, final_n_clusters = reorganize_clusters(df)

    # Create train/test splits
    df = create_train_test_split(df, 'cluster', 'kmeans_split')
    df = create_random_clusters(df, final_n_clusters)
    df = create_train_test_split(df, 'random_cluster', 'random_split')

    # Prepare final output, ensuring all cluster columns are integers
    df.rename(columns={'cluster': 'kmean_cluster'}, inplace=True)
    df['kmean_cluster'] = df['kmean_cluster'].astype(int)
    df['random_cluster'] = df['random_cluster'].astype(int)

    # Select only the required columns for the final output
    output_df = df[['huc_code', 'kmean_cluster', 'kmeans_split', 'random_cluster', 'random_split']]

    # Add excluded HUCs back to the output
    final_output_df = add_excluded_hucs_back(output_df, excluded_df)
    final_output_df['kmean_cluster'] = final_output_df['kmean_cluster'].astype(int)
    final_output_df['random_cluster'] = final_output_df['random_cluster'].astype(int)


    # Save final output
    output_file_path = f"/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_{final_n_clusters}_clusters_min_size_{min_cluster_size}.csv"
    final_output_df.to_csv(output_file_path, index=False)
    print(final_output_df.head())

    # Plotting
    plot_clusters(df, X, 'kmean_cluster', 'KMeans', f"/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/pca_kmeans_{final_n_clusters}_clusters_min_size_{min_cluster_size}.png")
    plot_clusters(df, X, 'random_cluster', 'Random', f"/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/pca_random_{final_n_clusters}_clusters_min_size_{min_cluster_size}.png")

    # Summary
    print("\nNumber of huc_code in each cluster, random cluster, and split:")
    print(final_output_df.groupby(['kmean_cluster', 'kmeans_split']).size())
    print(final_output_df.groupby(['random_cluster', 'random_split']).size())

# Run the script
if __name__ == "__main__":
    main()
