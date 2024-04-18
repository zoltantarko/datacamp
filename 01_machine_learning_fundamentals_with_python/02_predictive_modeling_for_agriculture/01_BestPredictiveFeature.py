from numpy import average
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

crops = pd.read_csv("../00_resources/00_resources/soil_measures.csv")
print(crops["crop"].unique())
print(crops.isna().sum().sort_values())

X = crops.drop("crop",axis=1)
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

feature_performance={}

for feature in ["N", "P", "K", "ph"]:
    logreg = LogisticRegression(multi_class="multinomial")
    logreg.fit(X_train[[feature]], y_train)
    y_pred = logreg.predict(X_test[[feature]])
    bas = metrics.balanced_accuracy_score(y_test, y_pred, adjusted=True)
    feature_performance[feature] = bas

max_value = max(feature_performance.values())
max_key = [key for key in feature_performance if feature_performance[key] == max_value]

best_predictive_feature = {str(max_key[0]): max_value}

print(best_predictive_feature)
