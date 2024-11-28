import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Feature Selection and Scaling
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def select_top_k_features(X, y, k=50):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    return selector.fit_transform(X, y)

# PCA for Dimensionality Reduction
def apply_pca(X, n_components=10):
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(X)
    return pca_fit, pca.transform(X)

def plot_scree(pca_fit):
    PC_values = np.arange(pca_fit.n_components_) + 1
    plt.figure(figsize=(8, 6))
    plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.grid()
    plt.savefig("results/DBSCAN/var_explained_mod2.png")

# k-Distance Plot for DBSCAN
def plot_k_distance(X_pca, k=4):
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    nearest_neighbors.fit(X_pca)
    distances, _ = nearest_neighbors.kneighbors(X_pca)
    distances = np.sort(distances[:, k - 1])
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title(f'{k}-Distance Graph')
    plt.xlabel('Data Points (sorted by distance)')
    plt.ylabel(f'{k}-Distance')
    plt.savefig("results/DBSCAN/k_explained_mod2.png")

# DBSCAN Clustering and Evaluation
def grid_search_dbscan(X_pca, eps_values, min_samples_values):
    best_eps, best_min_samples, best_score = None, None, -1
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_pca)
            
            if len(set(labels)) <= 1:
                continue  # Skip if only one cluster is formed
            
            score = silhouette_score(X_pca, labels)
            if score > best_score:
                best_eps, best_min_samples, best_score = eps, min_samples, score
    return best_eps, best_min_samples, best_score

def run_dbscan(X_pca, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_pca)
    
    if len(set(dbscan_labels)) > 1:  # Ensure there are at least 2 clusters
        sil_score = silhouette_score(X_pca, dbscan_labels)
        db_score = davies_bouldin_score(X_pca, dbscan_labels)
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f}")
    else:
        print("DBSCAN resulted in only one cluster. Consider adjusting eps or min_samples.")
    
    return dbscan_labels

def plot_dbscan_clusters(X_pca, dbscan_labels):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='Set1', s=50)
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.savefig("results/DBSCAN/DBSCAN_pca_mod2.png")


def analyze_cluster_counts(dbscan_labels):
    unique_labels, counts = np.unique(dbscan_labels, return_counts=True)
    print(f"Cluster counts: {dict(zip(unique_labels, counts))}")

# Main Analysis Pipeline
def main(X, y, k=50, n_components=10, eps_values=np.arange(1, 5, 0.5), min_samples_values=range(3, 10)):
    # Feature Selection and Scaling
    X_scaled = scale_features(X)
    X_selected = select_top_k_features(X_scaled, y, k)

    # PCA for Dimensionality Reduction
    pca_fit, X_pca = apply_pca(X_selected, n_components)

    # Plot Scree Plot
    plot_scree(pca_fit)

    # k-Distance Plot
    plot_k_distance(X_pca, k)

    # Grid Search for Best DBSCAN Parameters
    best_eps, best_min_samples, best_score = grid_search_dbscan(X_pca, eps_values, min_samples_values)
    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best Silhouette Score: {best_score:.3f}")

    # Run DBSCAN with Best Parameters
    dbscan_labels = run_dbscan(X_pca, best_eps, best_min_samples)

    # Plot DBSCAN Clusters
    plot_dbscan_clusters(X_pca, dbscan_labels)

    # Analyze Cluster Counts
    analyze_cluster_counts(dbscan_labels)

X=pd.read_csv("data/CIT_full_cleaned.csv",index_col="sample_id")
y=X["target"]
X=X.drop(columns="target")

main(X, y)
