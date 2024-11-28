import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import os

# Function to compute k-distance for DBSCAN parameter selection
def compute_k_distance(X, k):
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    nearest_neighbors.fit(X)
    distances, indices = nearest_neighbors.kneighbors(X)
    return np.sort(distances[:, k - 1])

# Load and preprocess data
X = pd.read_csv("data/CIT_full_cleaned.csv", index_col="sample_id")
X = X.drop(columns="target")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute k-distances for scaled and non-scaled data
k = 6
distances_non_scaled = compute_k_distance(X, k)
distances_scaled = compute_k_distance(X_scaled, k)

# Plot k-distance comparison
plt.figure(figsize=(8, 6))
plt.plot(distances_non_scaled, label='Non-Scaled', color='blue')
plt.plot(distances_scaled, label='Scaled', color='red')
plt.title('k-Distance Graph (Scaled vs Non-Scaled)')
plt.xlabel('Data Points (sorted by distance)')
plt.ylabel(f'{k}-Distance')
plt.legend()
plt.grid(True)
plt.savefig('results/DBSCAN/k_distance_comparison.png')

# PCA analysis
pca = PCA()
pca.fit(X_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance for Model 1')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.savefig("results/DBSCAN/cumulative_explained_var.png")

# Determine optimal number of components for desired variance
variance_threshold = 0.90
optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
print(f"Number of components to retain {variance_threshold * 100}% variance: {optimal_components}")

# Analyze DBSCAN clustering performance with different PCA components and parameters
best_score = -1
best_n_components = None
best_eps = None
best_min_samples = None

eps_values = np.arange(0.1, 1.5, 0.1)
min_samples_values = range(2, 10)

for n_components in range(1, optimal_components + 1):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_pca)

            # Skip if only one cluster is formed
            if len(set(labels)) <= 1:
                continue

            score = silhouette_score(X_pca, labels)
            if score > best_score:
                best_score = score
                best_n_components = n_components
                best_eps = eps
                best_min_samples = min_samples

# Print the best results
print(f"Best PCA components for Model 1: {best_n_components}")
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
print(f"Best Silhouette Score for Model 1: {best_score:.3f}")
