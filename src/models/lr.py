from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from joblib import dump, load

# Training model
def train_lr (X_train, X_test, y_train, y_test, feature_names):
      # Create logistic regression cross validation

      grid = {
            'C': np.power(10.0, np.arange(-10, 10)),
            'solver': ['saga'],
            'random_state': [0],
            'l1_ratio':np.arange(0, 1, 0.1)
            }
      
      clf = LogisticRegression(penalty='elasticnet', max_iter=100000)
      lg_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=10, n_jobs = 70)
      
      # Train the regressor
      lg_grid.fit(X_train, y_train)
      # Make predictions using the optimised parameters
      lg_pred = lg_grid.predict(X_test)
      # Save model
      dump(lg_grid, "../results/lg.joblib")
      # Get AUC
      roc_auc = round(roc_auc_score (y_test, lg_pred), 3)

      print("Roc_auc_score:", roc_auc)

      best_prs = lg_grid.best_estimator_

      print("Best Parameters:\n", best_prs)
      print("Best Score:\n", 'roc_auc:', roc_auc)
      # Get coefficients of the model 
      coef = pd.DataFrame(best_prs.coef_.T, index = feature_names)
      var = coef[coef[0] != 0]
      print("Head of coefficients", coef.sort_values(by = 0, ascending=False).head())
      print("Logistic (ElasticNet) picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
      var.to_csv('../results/log_features.csv')

     