import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

df=pd.read_csv("data/CIT_subtyping.csv")
df2=pd.read_csv("data/CIT_genesets.csv")
feats_xgboost=pd.read_csv("results/xgboost/feature_importance_xgb.csv")
feats_xgboost=feats_xgboost[feats_xgboost["Gain"]>0]
names_xgboost=feats_xgboost["Feature"].to_list()
df_genes_subtyping=df["Unnamed: 0"].to_list()
#with open("CIT_genes_subtyping.txt","w") as file:
#    for i in df_genes_subtyping:
#        file.write(i+"\n")
#    file.close()
all_genes=set(df2["Genes"].to_list())
counts=0
for i in names_xgboost:
    if i in df_genes_subtyping:
        print(i)
        counts+=1

print(len(df_genes_subtyping))
print(len(names_xgboost))
print(counts)
# Convert data to sets
# Convert data to sets
set1 = set(df_genes_subtyping)
set2 = set(names_xgboost)
plt.figure(figsize=(10, 8))  # Increase the size (width, height)

# Create the Venn diagram
venn = venn2([set1, set2], ('Known Subtyping Genes', 'XGBoost Genes'))

# Customize the labels for the sets
for label in venn.set_labels:
    if label:  # Check if the label exists
        label.set_fontsize(12)

# Customize the subset labels
for subset in venn.subset_labels:
    if subset:  # Check if there's a label to set
        subset.set_fontsize(10)

# Add a title
plt.title("Intersections between XGboost relevant features and \n Subtyping reported genes", fontsize=14)

# Save the plot to a file
plt.savefig("venn_diagram.png", dpi=300)

# Optionally display the plot