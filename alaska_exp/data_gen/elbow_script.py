import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the data
file_path = 'average_values_cleaned_.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Drop `huc_code` for clustering
features = data.drop(columns=['huc_code'])

# Handle missing values by imputing the mean
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Check for any remaining NaNs
assert not features.isna().any().any(), "There are still NaNs in the data."

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans for different numbers of clusters
wcss = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, wcss, marker='o', linestyle='--')
plt.title('Optimal Clusters of Watersheds')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)

# Save the plot as a PNG file
plot_path = 'elbow_method_plot.png'  # Specify the file name and path
plt.savefig(plot_path, dpi=600, bbox_inches='tight')  # Save with high resolution
plt.show()

print(f"Elbow method plot saved to {plot_path}")