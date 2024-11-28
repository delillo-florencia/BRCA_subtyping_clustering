import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("data/CIT_full.csv")
df_classes=pd.read_csv("data/CIT_classes.csv")


df_gene_expression = df.transpose()
df_gene_expression.columns = df_gene_expression.iloc[0]  # Set the first row as column headers
df_transposed = df_gene_expression.drop(df_gene_expression.index[0]) 
df_classes=df_classes.rename(columns={"Unnamed: 0	":"sample_id","x":"target"})
df_transposed=df_transposed.rename(columns={"Unnamed: 0	":"sample_id"})

print(df_transposed.head())
print(df_classes.head())
df = df_transposed.merge(df_classes, left_index=True, right_on='Unnamed: 0')
df=df.rename(columns={'Unnamed: 0':"sample_id"})
df=df.set_index("sample_id")

df.to_csv("CIT_full_cleaned.csv",index=True)
X = df.drop(['target'], axis=1)  # Features (expression data)
X = X.apply(pd.to_numeric, errors='coerce')
X.columns = [f'{col}_{i}' if X.columns.tolist().count(col) > 1 else col for i, col in enumerate(X.columns)]
y = df['target']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

print("Training set class distribution:")
print(y_train.value_counts(normalize=True))
print("Test set class distribution:")
print(y_test.value_counts(normalize=True))

X_train.to_csv("data/X_train.csv")
X_test.to_csv("data/X_test.csv")
y_train.to_csv("data/y_train.csv")
y_test.to_csv("data/y_test.csv")
