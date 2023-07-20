
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer
from sklearn import svm
import pandas as pd
import numpy as np

# Select features using decision tree model
def rfe_svm(X_train, y_train, X_test, y_test):
    rfe_svm = svm.LinearSVC()
    best_auc = list()
    all_scores = []
    features = []

    for i in range(1, len(X_train[0])):
        rfe = RFE(rfe_svm, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)

        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)

        # Decision tree model
        acr, pre_score, rc_score, f1, roc_auc = train_svm(X_train_reduce, y_train, X_test_reduce, y_test)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        all_scores.append([acr, pre_score, rc_score, f1, roc_auc])

    pd.DataFrame({'features':features}).to_csv('svm_features.csv')
    pd.DataFrame({'all_score':all_scores}).to_csv('svm_all_scores.csv')
    print("The best AUC of SVM: ", max(best_auc))


def train_svm(X_train, y_train, X_test, y_test):

    # Create svm cross validation

    grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    
    clf = svm.SVC()
    svm_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5)
    
    # Train the regressor
    svm_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    svm_pred = svm_grid.predict(X_test)
    
    # Find scores 
    acr = accuracy_score(y_test, svm_pred)
    f1 = f1_score (y_test, svm_pred)
    pre_score = precision_score(y_test, svm_pred)
    rc_score = recall_score (y_test, svm_pred)

    roc_auc = round(roc_auc_score (y_test, svm_pred), 3)

    #print("accuracy:", acr,  "precision_score:", 
            #pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)
    #print("accuracy:", acr)
    return (acr, pre_score, rc_score, f1, roc_auc)