import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
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

# Directory where your CSV files are located
folder_path = 'results/xgboost/'  # Change to your folder path

# Read and process each CSV file
itemsets_by_subtype = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith('itemsets.csv'):
        subtype = file_name.split('.')[0]  # Assuming the file name is the subtype name
        file_path = os.path.join(folder_path, file_name)
        itemsets_by_subtype[subtype] = read_and_filter_csv(file_path)

# Find unique itemsets for each subtype
unique_itemsets = find_unique_itemsets(itemsets_by_subtype)

# Print results
for subtype, unique_sets in unique_itemsets.items():
    print(f"Unique itemsets for {subtype}: {unique_sets}")
# Find unique itemsets for each subtype
unique_itemsets = find_unique_itemsets(itemsets_by_subtype)

# Create a DataFrame to display the results as a table
unique_items_table = pd.DataFrame({
    'Subtype': list(unique_itemsets.keys()),
    'Unique Itemsets': [len(itemsets) for itemsets in unique_itemsets.values()]
})

# Display the table
print("Unique Itemsets Table:")
print(unique_items_table)

# Plot the number of unique itemsets for each subtype
plt.figure(figsize=(10, 6))
plt.bar(unique_items_table['Subtype'], unique_items_table['Unique Itemsets'], color='skyblue')
plt.xlabel('Subtype')
plt.ylabel('Number of Unique Itemsets')
plt.title('Number of Unique Itemsets per Subtype')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.savefig("results/analysis_plots/unique_freq_items.png")