import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns

subtypes=pd.read_csv("data/CIT_classes_cleaned.csv")
top_genes=pd.read_csv("results/xgboost/feature_importance_xgb.csv")

df = pd.read_csv("data/CIT_full_cleaned.csv")  
df_merge=pd.merge(df,subtypes,on="sample_id")

filtered_top_genes=top_genes[top_genes["Gain"]>0]
sorted_filtered_genes=filtered_top_genes.sort_values(by="Gain")[0:30]["Feature"].to_list()
sorted_filtered_genes_w_subtype=["subtype"]+sorted_filtered_genes

data=df_merge[sorted_filtered_genes_w_subtype]

threshold = data.median(axis=0)  # Median expression for each gene

binary_data = data.copy()  
binary_data[sorted_filtered_genes] = data[sorted_filtered_genes].gt(threshold, axis=1).astype(int)

subtype_groups = binary_data.groupby("subtype")

frequent_itemsets_by_subtype = {}

for subtype, group in subtype_groups:
    print(subtype)

    gene_data = group.drop(columns=["subtype"])
    
    frequent_itemsets = fpgrowth(gene_data, min_support=0.1, use_colnames=True)
    frequent_itemsets_by_subtype[subtype] = frequent_itemsets
    frequent_itemsets_sorted=frequent_itemsets.sort_values(by="support",ascending=False)

    output_file =  f"results/xgboost/{subtype}_frequent_itemsets.csv"
    frequent_itemsets_sorted.to_csv(output_file, index=False)


# Output results
for subtype, itemsets in frequent_itemsets_by_subtype.items():
    print(f"Frequent itemsets for subtype {subtype}:")
    print(itemsets)
    print("\n")

# Collecting the number of frequent itemsets for each subtype
subtype_counts = {subtype: len(itemsets) for subtype, itemsets in frequent_itemsets_by_subtype.items()}

# Convert to DataFrame for easier plotting with seaborn
subtype_counts_df = pd.DataFrame(list(subtype_counts.items()), columns=["Subtype", "Frequent Itemset Count"])

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Subtype", y="Frequent Itemset Count", data=subtype_counts_df, palette="viridis")
plt.title("Number of Frequent Itemsets per Subtype")
plt.xlabel("Subtype")
plt.ylabel("Number of Frequent Itemsets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/xgboost/frequent_items_subtype.png")

