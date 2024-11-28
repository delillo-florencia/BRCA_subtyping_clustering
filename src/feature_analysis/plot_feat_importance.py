import pandas as pd
import matplotlib.pyplot as plt
from gprofiler import GProfiler
def check_cancer_association(features):
    cancer_genes = []
    for gene in features:
        res = mg.query(gene, scopes="symbol", fields="disease", species="human")
        if res and "hits" in res:
            for hit in res["hits"]:
                if "disease" in hit and "breast_cancer" in hit["disease"].lower():
                    cancer_genes.append(gene)
                    break
    return cancer_genes
df = pd.read_csv("results/xgboost/feature_importance_xgb.csv")

# Sort by Gain and select top 10 features
top_10 = df.nlargest(10, 'Gain')
relevant=df[df["Gain"]>0]
print(top_10)
# Plotting top 10 features
plt.figure(figsize=(10, 6))
plt.barh(top_10['Feature'], top_10['Gain'], color='skyblue')
plt.xlabel('Gain')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("results/analysis_plots/top_gain_features.png")

# Gene Ontology analysis
# Initialize the GO profiler
gp = GProfiler(return_dataframe=True)

# Perform GO analysis
results = gp.profile(
    organism='hsapiens',  # Specify organism (human in this case)
    query=relevant['Feature'].tolist()
)
print(results.columns)
# Show the first few results
results.to_csv("results/analysis_plots/analysis_plots.csv",index=False)

