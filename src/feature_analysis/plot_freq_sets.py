import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict  # Import defaultdict

# Function to read and filter CSV file by support > 0.7
def read_and_filter_csv(file_path):
    df = pd.read_csv(file_path)
    df_filtered = df[df['support'] > 0.7]
    df_filtered['itemsets'] = df_filtered['itemsets'].apply(eval)  # Convert frozenset from string to actual frozenset
    return df_filtered['itemsets'].tolist()

# Function to find unique itemsets for each subtype
def find_unique_itemsets(itemsets_by_subtype):
    all_itemsets = set()  # To track all itemsets across all subtypes
    unique_itemsets = defaultdict(set)  # Dictionary to store unique itemsets for each subtype
    
    # Collect all itemsets across subtypes
    for subtype, itemsets in itemsets_by_subtype.items():
        all_itemsets.update(itemsets)
    
    # Identify unique itemsets for each subtype
    for subtype, itemsets in itemsets_by_subtype.items():
        # Unique itemsets for this subtype (those that are not in other subtypes)
        unique_itemsets[subtype] = set(itemsets) - (all_itemsets - set(itemsets))
    
    return unique_itemsets

# Function to plot heatmap and boxplots for gene expression across subtypes
def plot_gene_expression_heatmap_and_boxplots(gene_expression_df, unique_itemsets):
    # Plot heatmap for each subtype with all genes
    for subtype in unique_itemsets.keys():
        # Subset the gene expression data to the current subtype
        subtype_data = gene_expression_df[gene_expression_df['subtype'] == subtype]
        
        # Drop the 'subtype' column to focus on the genes
        subtype_expression_data = subtype_data.drop(columns=['subtype', 'sample_id'])

        # Plot heatmap for this subtype
        plt.figure(figsize=(12, 6))
        sns.heatmap(subtype_expression_data, cmap='coolwarm', annot=False, cbar_kws={'label': 'Expression Level'})
        plt.title(f"Heatmap of Gene Expression for {subtype}")
        plt.xlabel('Genes')
        plt.ylabel('Samples')
        plt.tight_layout()
        plt.savefig("results/analysis_plots/heatmap.png")

        # Plot boxplot for each gene across the subtype
        for gene in subtype_expression_data.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=subtype_expression_data[gene], color='lightblue')
            plt.title(f"Boxplot of {gene} Expression in {subtype}")
            plt.xlabel('Expression Level')
            plt.ylabel('Gene Expression')
            plt.tight_layout()
            plt.show()

# Directory where your CSV files are located
folder_path = 'results/xgboost/'  # Change to your folder path

# Read and process each CSV file for frequent itemsets
itemsets_by_subtype = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith('itemsets.csv'):
        subtype = file_name.split('.')[0]  # Assuming the file name is the subtype name
        file_path = os.path.join(folder_path, file_name)
        itemsets_by_subtype[subtype] = read_and_filter_csv(file_path)

# Find unique itemsets for each subtype
unique_itemsets = find_unique_itemsets(itemsets_by_subtype)

# Load the gene expression data (assuming it is in a CSV file)
gene_expression_path = 'data/CIT_full_cleaned.csv'  # Change this to the actual path
gene_expression_df = pd.read_csv(gene_expression_path)
gene_expression_df=gene_expression_df.rename(columns={"target":"subtype"})
# Plot heatmap and boxplots for gene expression across subtypes
plot_gene_expression_heatmap_and_boxplots(gene_expression_df, unique_itemsets)
