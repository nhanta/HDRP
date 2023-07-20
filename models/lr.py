from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer
import pandas as pd
import numpy as np

# Training model
def train_lr (X_train, X_test, y_train, y_test, feature_names):
  # Create logistic regression cross validation

  grid = {
        'C': np.power(10.0, np.arange(-5, 5)),
         'solver': ['saga'],
          'random_state': [0],
          'l1_ratio':[0.1,0.3,0.5,0.8]
        }
  
  clf = LogisticRegression(penalty='elasticnet', random_state=0, max_iter=10000, n_jobs = 10)
  lg_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5)
  
  # Train the regressor
  lg_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  lg_pred = lg_grid.predict(X_test)

  
  # Find scores 
  acr = accuracy_score(y_test, lg_pred)
  f1 = f1_score (y_test, lg_pred)
  pre_score = precision_score(y_test, lg_pred)
  rc_score = recall_score (y_test, lg_pred)

  roc_auc = round(roc_auc_score (y_test, lg_pred), 3)

  print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)

  best_prs = lg_grid.best_estimator_

  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'roc_auc:', roc_auc)
  # Get coefficients of the model 
  coef = pd.DataFrame(best_prs.coef_.T, index = feature_names)
  var = coef[coef[0] != 0]
  print("Head of coefficients", coef.sort_values(by = 0, ascending=False).head())
  print("ElasticNet picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  #pd.DataFrame({'variable':var}).to_csv('log_features.csv')
  return (acr, pre_score, rc_score, f1, roc_auc)

