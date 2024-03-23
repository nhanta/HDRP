from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer
import pandas as pd
import numpy as np

# Select features using decision tree model
def rfe_dt(X_train, y_train, X_test, y_test):
    rfe_dt = tree.DecisionTreeClassifier()
    best_auc = list()
    all_scores = []
    features = []

    for i in range(1, len(X_train[0])):
        rfe = RFE(rfe_dt, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)

        # Decision tree model
        acr, pre_score, rc_score, f1, roc_auc = train_dt(X_train_reduce, y_train, X_test_reduce, y_test)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        all_scores.append([acr, pre_score, rc_score, f1, roc_auc])

    pd.DataFrame({'features':features}).to_csv('dt_features.csv')
    pd.DataFrame({'all_score':all_scores}).to_csv('dt_all_scores.csv')
    print("The best AUC of Decision Tree: ", max(best_auc))

def train_dt(X_train, y_train, X_test, y_test):

    # Create decision-tree cross validation

    grid = {'criterion': ["gini", "entropy", "log_loss"], 
            'max_features': ["sqrt","log2"]}
    
    clf = tree.DecisionTreeClassifier()
    dt_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5)
    
    # Train the regressor
    dt_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    dt_pred = dt_grid.predict(X_test)
    
    # Find scores 
    acr = accuracy_score(y_test, dt_pred)
    f1 = f1_score (y_test, dt_pred)
    pre_score = precision_score(y_test, dt_pred)
    rc_score = recall_score (y_test, dt_pred)

    roc_auc = round(roc_auc_score (y_test, dt_pred), 3)

    #print("accuracy:", acr,  "precision_score:", 
            #pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)
    #print("accuracy:", acr)
    return (acr, pre_score, rc_score, f1, roc_auc)