import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

gene_expression_df=pd.read_csv("data/CIT_full_cleaned.csv")
gene_expression_df=gene_expression_df.rename(columns={"target":"subtype"})
df = pd.read_csv("results/xgboost/feature_importance_xgb.csv")

# Sort by Gain and select top 10 features
top_genes = df.nlargest(10, 'Gain')["Feature"].to_list()


# Create a DataFrame for plotting with only the top 10 genes and their corresponding subtypes
top_genes_df = gene_expression_df[['subtype'] + top_genes]

# Melt the DataFrame to make it long-form for easier plotting with seaborn
top_genes_long = pd.melt(top_genes_df, id_vars=['subtype'], value_vars=top_genes, 
                         var_name='gene', value_name='expression')

# Plot gene expression for top 10 genes per subtype
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_genes_long, x='gene', y='expression', hue='subtype', palette='Set2')

# Customize plot
plt.title('Gene Expression of Top 10 Genes per Subtype')
plt.xticks(rotation=45)
plt.xlabel('Gene')
plt.ylabel('Expression')
plt.legend(title='Subtype', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

plt.tight_layout()
plt.savefig("results/analysis_plots/expression.png")
