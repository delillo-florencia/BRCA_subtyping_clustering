import pandas as pd
import networkx as nx


# Step 1: Load the precomputed graphs from the .graphml file
graph_files = ["sig_genes_graph_pearson.graphml",
               "sig_genes_graph_cosine.graphml"] # Replace with your .graphml file path

pearson_graph = nx.read_graphml(graph_files[0])
cosine_graph = nx.read_graphml(graph_files[1])

# Step 2: Apply Louvain clustering
print("Running Louvain clustering...")
communities_pearson = nx.community.louvain_communities(pearson_graph, seed=123)  # Seed for reproducibility
communities_cosine = nx.community.louvain_communities(cosine_graph, seed=123)  # Seed for reproducibility

# Step 3: Output the number of communities detected
print(f"Number of communities detected using pearson matrix: {len(communities_pearson)}")
print(f"Number of communities detected using cosine matrix: {len(communities_cosine)}")

# Step 4: Save the communities to a file for later inspection
output_files = ["sig_louvain_communities_pearson.txt",
                "sig_louvain_communities_cosine.txt"]

print(f"Saving communities to {output_files[0]}...")

with open(output_files[0], "w") as f:
    for i, community in enumerate(communities_pearson):
        f.write(f"Community {i} ({len(community)} nodes): {', '.join(community)}\n")


print(f"Saving communities to {output_files[1]}...")

with open(output_files[1], "w") as f:
    for i, community in enumerate(communities_cosine):
        f.write(f"Community {i} ({len(community)} nodes): {', '.join(community)}\n")





print("Louvain clustering completed and results saved.")