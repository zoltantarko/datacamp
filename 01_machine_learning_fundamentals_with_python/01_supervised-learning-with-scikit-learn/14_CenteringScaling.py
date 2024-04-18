# Import StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

music_df = pd.read_csv("../music_clean.csv", header=0)
#X = music_df.drop("loudness", axis=1).values
#y = music_df["loudness"].values
X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline steps
#steps = [("scaler", StandardScaler()),
#         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
#pipeline = Pipeline(steps)
#pipeline.fit(X_train, y_train)

# Calculate and print R-squared
#print(pipeline.score(X_test, y_test))

steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)