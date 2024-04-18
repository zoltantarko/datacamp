# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd

X_new = pd.DataFrame({'account_length': [30.0, 107.0, 213.0], 'customer_service_calls': [17.5, 24.1, 10.9]})

churn_df = pd.read_csv('../00_resources/churn_df.csv', header=0)
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X, y)
y_pred = knn.predict(X_new)
print("Predictions: {}".format(y_pred)) 