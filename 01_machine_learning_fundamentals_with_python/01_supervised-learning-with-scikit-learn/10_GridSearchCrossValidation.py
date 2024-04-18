from sklearn.model_selection import GridSearchCV, train_test_split, KFold
import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd


diabetes_df = pd.read_csv('../00_resources/diabetes_clean.csv', header=0)
# Create X and y arrays
X = diabetes_df.drop("diabetes", axis=1).values
y = diabetes_df["diabetes"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=5)
# Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}
lasso = Lasso()

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))