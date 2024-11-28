import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# 1. Load important features from xgb_model
features = pd.read_csv('results/xgboost/feature_importance_xgb.csv')
features = features[(features['Gain'] != 0) & (features['Weight'] != 0)]

# 2. Load the complete dataset
full_df = pd.read_csv('data/CIT_full_cleaned.csv').T

# 3. Filter significant genes from the dataset based on xgb model
sig_genes = list(features['Feature'])
sig_genes_df = full_df.loc[sig_genes]

# 4. Calculate correlation and similarity matrices
pearson_matrix = sig_genes_df.T.corr(method='pearson')
pearson_matrix_df = pd.DataFrame(pearson_matrix, index=sig_genes_df.index, columns=sig_genes_df.index)

cosine_matrix = cosine_similarity(sig_genes_df)
cosine_matrix_df = pd.DataFrame(cosine_matrix, index=sig_genes_df.index, columns=sig_genes_df.index)

# Display correlation and similarity matrices
print("Cosine Similarity Matrix:")
print(cosine_matrix_df)
print("Pearson Correlation Matrix:")
print(pearson_matrix_df)

# 5. Generate graphs and save them
graph_pearson = nx.from_pandas_adjacency(pearson_matrix_df)
nx.write_graphml(graph_pearson, "results/graph_clustering/sig_genes_graph_pearson.graphml")

graph_cosine = nx.from_pandas_adjacency(cosine_matrix_df)
nx.write_graphml(graph_cosine, "results/graph_clustering/sig_genes_graph_cosine.graphml")

# Optional: Visualize the cosine similarity graph
plt.figure(figsize=(12, 12))
nx.draw(graph_cosine, with_labels=False, node_size=2, font_size=1)
plt.show()

# Placeholder for additional functionality (e.g., clustering)
raise NotImplementedError("Clustering and further analysis not yet implemented.")
