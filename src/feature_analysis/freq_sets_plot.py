import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load gene expression data
gene_expression = pd.read_csv('data/CIT_full_cleaned.csv')  # Example file
df=gene_expression.rename(columns={"target":"subtype"})

gene_sets = {
    'basL': ['LGR6', 'KRT5', 'KRT17', 'MUC15', 'MET', 'WFDC21P', 'MRPL47'],
    'lumA': ['ZNF205-AS1', 'UGT2B28', 'CCDC94', 'PIFO', 'PCAT18'],
    'lumB': ['TOB1'],
    'lumC': ['IL1R1'],
    'mApo': ['S100A10', 'WFDC21P', 'MUC15', 'UGT2B28'],
    'normL': ['KRT5', 'TPCN1', 'HNRNPA3', 'IL1R1', 'F2R', 'LGR6']
}


# Create a plot for each gene set
sns.set(style="whitegrid")
for gene_set_name, genes in gene_sets.items():
    plt.figure(figsize=(12, 6))
    
    # Prepare data for the plot: melt gene expression data for the current gene set
    df_subset = df[['subtype'] + genes].melt(id_vars=['subtype'], value_vars=genes, var_name='Gene', value_name='Expression')
    
    # Plot the gene expressions for the gene set across subtypes
    sns.boxplot(x='subtype', y='Expression', hue='Gene', data=df_subset, palette='Set2')
    plt.title(f'Gene Expression across Subtypes for {gene_set_name} freq_sets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/analysis_plots/boxplot_{gene_set_name}_freq.png")


# Convert data to DataFrame

# List of all genes across all subtypes
all_genes = ['LGR6', 'KRT5', 'KRT17', 'MUC15', 'MET', 'WFDC21P', 'MRPL47',
             'ZNF205-AS1', 'UGT2B28', 'CCDC94', 'PIFO', 'PCAT18', 'TOB1', 'IL1R1',
             'S100A10', 'TPCN1', 'HNRNPA3', 'F2R']

# Extract the relevant gene expression data for all genes
df_subset = df[['subtype'] + all_genes]

# Pivot the data so that genes are rows and subtypes are columns
df_pivoted = df_subset.set_index('subtype').T

# Plot heatmap for the gene expression across subtypes
plt.figure(figsize=(12, 8))
sns.heatmap(df_pivoted, annot=True, cmap='viridis', cbar_kws={'label': 'Gene Expression'}, linewidths=0.5)
plt.title('Gene Expression Heatmap Across All Subtypes')
plt.xlabel('Subtype')
plt.ylabel('Gene')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("image.png")