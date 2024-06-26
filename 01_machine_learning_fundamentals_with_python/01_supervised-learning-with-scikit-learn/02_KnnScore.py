# Import the module
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd


churn_df = pd.read_csv('../00_resources/churn_df.csv', header=0)

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))