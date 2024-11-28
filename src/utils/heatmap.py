import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("data/CIT_full_cleaned.csv")
df=df.rename(columns={"target":"subtype"})
gene_list= ['LGR6', 'KRT5', 'KRT17', 'MUC15', 'MET', 'WFDC21P', 'MRPL47',
             'ZNF205-AS1', 'UGT2B28', 'CCDC94', 'PIFO', 'PCAT18', 'TOB1', 'IL1R1',
             'S100A10', 'TPCN1', 'HNRNPA3', 'F2R']
# Select the columns needed for the heatmap
gene_list = ["sample_id", "subtype"] + gene_list
df_merged = df[gene_list]

# Set the 'sample' column as the index
df_merged.set_index('sample_id', inplace=True)

# Calculate global min and max for the color scale
gene_data = df_merged.drop(columns=['subtype'])  # Exclude the 'subtype' column
vmin, vmax = gene_data.min().min(), gene_data.max().max()

# Create heatmaps for each subtype
subtypes = df_merged['subtype'].unique()

for subtype in subtypes:
    # Filter the data for the current subtype
    df_subtype = df_merged[df_merged['subtype'] == subtype]
    
    # Drop the 'subtype' column since it's not part of the heatmap
    df_subtype = df_subtype.drop(columns=['subtype'])
    
    # Create the heatmap with consistent color scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_subtype.T, cmap="viridis", cbar=True, annot=False, vmin=vmin, vmax=vmax)
    plt.title(f"Heatmap of Gene Expressions for Subtype {subtype}")
    plt.xlabel("Samples")
    plt.ylabel("Genes")
    
    # Show the plot
    plt.tight_layout()

    plt.savefig(f"results/analysis_plots/boxplot_{subtype}.png")

'''
file_path = "results/GE_purity/top_10_genes.txt"
# Open the file and read the lines into a list
with open(file_path, 'r') as file:
    gene_list = [line.strip() for line in file.readlines()]

# Read the labels and expression data
df_labels = pd.read_csv("cleaned_data/subclasses.csv")
df_expression = pd.read_csv("cleaned_data/gene_Expression.csv")

df_merged=pd.merge(df_labels,df_expression,on="sample")

gene_list=["sample","subtype"]+gene_list

df_merged=df_merged[gene_list]

# Step 3: Set the 'sample' column as the index (if it's not already)
df_merged.set_index('sample', inplace=True)

# Step 4: Normalize the expression data (optional but recommended)
# You can use StandardScaler to scale the expression values (mean = 0, std = 1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_merged.drop(columns=['subtype'])), columns=df_merged.columns[:-1])

# Step 5: Create the heatmap
# Ensure we're plotting the right data (df_scaled) and not including 'subclass' in the heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(df_scaled.T, cmap='coolwarm', annot=False, xticklabels=df_scaled.columns, yticklabels=df_scaled.index)

# Add title and labels
plt.title("Gene Expression Heatmap by Cancer Subtypes", fontsize=16)
plt.xlabel("Sample", fontsize=12)
plt.ylabel("Gene", fontsize=12)

plt.show()
'''