from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from joblib import dump

# Select features using decision tree model
def rfe_dt(X_train, y_train, X_test, y_test):
    
    rfe_dt = tree.DecisionTreeClassifier(random_state=7)
    best_auc = list()
    features = []
    iddd = []
    gr = []

    for i in range(1, len(X_train[0])):
        rfe = RFE(rfe_dt, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)
        # Decision tree model
        roc_auc, dt_grid = train_dt(X_train_reduce, y_train, X_test_reduce, y_test)
        print(i, roc_auc)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        gr.append(dt_grid)
        iddd.append(i)

    print("The best AUC of Decision Tree: ", max(best_auc))
    idd = np.argmax(best_auc)
    print("Number of Selected Features is: ", iddd[idd])
    ft = features[idd] 

    dump(gr[idd], "../results/t.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv('../results/dt_features.csv')


def train_dt(X_train, y_train, X_test, y_test):

    # Create decision-tree cross validation

    grid = {'criterion': ["gini", "entropy"], 
            'splitter': ["best", "random"],
            'max_features': ["auto", "sqrt","log2"],
            'max_depth' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
    
    clf = tree.DecisionTreeClassifier(random_state=7)
    dt_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    dt_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    dt_pred = dt_grid.predict(X_test)
    roc_auc = round(roc_auc_score (y_test, dt_pred), 3)

    return (roc_auc, dt_grid)