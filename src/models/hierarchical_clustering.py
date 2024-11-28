import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA 
import scipy.cluster.hierarchy as sch
import heapq


## Functions:

# PCA for Dimensionality Reduction
def apply_pca(X, n_components=10):
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(X)
    explained_variance = pca.explained_variance_ratio_

    return pca_fit,explained_variance, pca.transform(X), pca.components_


def hierarchical_clustering(pca_df):

    X = pca_df.to_numpy()
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance_matrix[i, j] = np.linalg.norm([X[i], X[j]])
            distance_matrix[j, i] = distance_matrix[i, j]

    linkage_matrix = linkage(X, method='ward', metric='euclidean')

    return n_samples, linkage_matrix


def pca_analisis(pcn, df_clust, n_component = 0):

    #pc1 = pcn[n_component]
    smallest_values_with_indices = heapq.nsmallest(10, enumerate(pcn[n_component]), key=lambda x: x[1])
    smallest_values = [x[1] for x in smallest_values_with_indices]
    smallest_indices = [x[0] for x in smallest_values_with_indices]

    print(f'Most important variables for the negative direction on pc{n_component+1}')
    print(df_clust.iloc[:, smallest_indices].columns)

    largest_values_with_indices = heapq.nlargest(10, enumerate(pcn[n_component]), key=lambda x: x[1])
    largest_values = [x[1] for x in largest_values_with_indices]
    largest_indices = [x[0] for x in largest_values_with_indices]

    print(f'Most important variables for the positive direction on pc{n_component+1}')
    print(df_clust.iloc[:, largest_indices].columns)

def pca_ranking(df_clust, loadings, top_k = None):
    # Example dataset feature names
    features = df_clust.columns

    # Loadings matrix (rows = PCs, columns = features)
    #loadings = pca.components_

    # Specify the number of top features to extract
    if top_k == None:
        top_k = loadings.shape[1]

    # Find top features for each component
    top_features_per_pc = {}

    for i, pc in enumerate(loadings):
        # Get absolute loadings and sort by importance
        sorted_indices = np.argsort(np.abs(pc))[::-1][:top_k]  # Top-k largest (absolute) loadings
        top_features_per_pc[f'PC{i+1}'] = features[sorted_indices]

    # Convert to a DataFrame for better visualization
    top_features_df = pd.DataFrame(top_features_per_pc)
    print(top_features_df)


## Plots

def plot_scree(pca_fit):
    PC_values = np.arange(pca_fit.n_components_) + 1
    plt.figure(figsize=(8, 6))
    plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.grid()
    plt.savefig("../../results/hierarchical_clustering/var_explained_pca10.png")



def plot_dendogram(n_samples, linkage_matrix):

    plt.figure(figsize=(20, 6))
    sch.dendrogram(linkage_matrix, labels=[f"Point {i}" for i in range(n_samples)])
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.grid()
    plt.savefig("../../results/hierarchical_clustering/hierarchichal_dendogram.png")

def plot_cluster_results(pca_df, linkage_matrix, num_clusters = 6 ):

    X = pca_df.to_numpy()
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title("Hierarchical Clustering Results (Scatter Plot)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid()
    plt.savefig("../../results/hierarchical_clustering/pca_cluster_results.png")


def main():

    ## Loading data:
    df=pd.read_csv("../../data/CIT_full.csv")
    df_classes=pd.read_csv("../../data/CIT_classes.csv")

    df_clust = df.copy()
    columns_names = df_clust['Unnamed: 0'].to_numpy()
    df_clust.drop(columns='Unnamed: 0', inplace=True)
    df_clust = df_clust.T
    df_clust = df_clust.set_axis(columns_names, axis=1)

    # Check contribution of first 10 components
    pca_fit_10, exp_variance , _ , _= apply_pca(df_clust)
    plot_scree(pca_fit_10)
    print(f"Explained variance ratio: {exp_variance}")

    # We keep the first 7 cause they are the ones explaining most of the variance
    pca_fit_7, exp_variance7, p_components, loadings = apply_pca(X = df_clust, n_components = 7)

    # PCA dataframe
    pca_colnames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
    pca_df = pd.DataFrame(data = p_components, columns = pca_colnames)

    # Hierarichal clustering:
    n_samples, ln_matrix = hierarchical_clustering(pca_df = pca_df)
    plot_dendogram(n_samples = n_samples, linkage_matrix = ln_matrix)

    # Analysis:
    pca_analisis(df_clust = df_clust, pcn = loadings, n_component=0)
    pca_analisis(df_clust = df_clust, pcn = loadings, n_component=1)
    pca_ranking(df_clust = df_clust, loadings = loadings, top_k = 670)

main()