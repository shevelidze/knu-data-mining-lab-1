import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with actual dataset path or URL)
data = pd.read_csv("Groceries_dataset.csv")


# Data Preprocessing: Transform transactions into a suitable format
def transform_data(data):
    transactions = data.groupby("Member_number")["itemDescription"].apply(list).tolist()
    unique_items = set(item for sublist in transactions for item in sublist)
    one_hot_encoded = pd.DataFrame(
        [
            {item: (item in transaction) for item in unique_items}
            for transaction in transactions
        ]
    )
    return one_hot_encoded


basket = transform_data(data)

# Exploratory Data Analysis: Most frequent items
item_counts = basket.sum().sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 5))
sns.barplot(x=item_counts.index, y=item_counts.values)
plt.xticks(rotation=45)
plt.title("Top 10 Most Frequent Items")
plt.show()

# Applying Apriori Algorithm
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filtering significant rules
strong_rules = rules[(rules["confidence"] > 0.5) & (rules["lift"] > 1.2)]
print(strong_rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# Visualizing the association rules
plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=strong_rules["support"],
    y=strong_rules["confidence"],
    size=strong_rules["lift"],
    hue=strong_rules["lift"],
    palette="coolwarm",
)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules (Support vs Confidence)")
plt.show()
