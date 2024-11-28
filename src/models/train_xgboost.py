import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

X_train=pd.read_csv("data/X_train.csv",index_col="sample_id")
X_test=pd.read_csv("data/X_test.csv",index_col="sample_id")
y_train=pd.read_csv("data/y_train.csv",index_col="sample_id")
y_test=pd.read_csv("data/y_test.csv",index_col="sample_id")

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

clf = xgb.XGBClassifier(
    objective='multi:softmax',   
    num_class=len(encoder.classes_), 
    eval_metric='mlogloss',  
    max_depth=6,  
    learning_rate=0.1, 
    n_estimators=100,  
    silent=1  
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))

xgb.plot_importance(clf)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_report = classification_report(y_test, y_pred)
print(class_report)

with open("results/classification_report_xgb.txt", "w") as file:
    file.write(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_names = list(encoder.classes_)  

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix_xgb.png")

# Feature importance
gain_importance = clf.get_booster().get_score(importance_type='gain')
weight_importance = clf.get_booster().get_score(importance_type='weight')

features = X_train.columns
gain_importance = {f: gain_importance.get(f, 0) for f in features}
weight_importance = {f: weight_importance.get(f, 0) for f in features}

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Gain': [gain_importance[f] for f in features],
    'Weight': [weight_importance[f] for f in features]
}).sort_values(by='Gain', ascending=False)

feature_importance_df.to_csv("results/xgboost/feature_importance_xgb.csv", index=False)
