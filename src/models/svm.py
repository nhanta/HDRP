
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.svm import SVC
from joblib import dump
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Select features using decision tree model
def rfe_svc(X_train, y_train, X_test, y_test):
    clf = SVC(kernel="linear", random_state=7)
    best_auc = list()
    features = []
    iddd = []
    gr = []
    l = len(X_train[0])
    
    for i in range(1, l):
        rfe = RFE(estimator=clf, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)
        
        # Decision tree model
        roc_auc, grid = train_svc(X_train_reduce, y_train, X_test_reduce, y_test)
        print("Number of Selected Features", i, "AUC", roc_auc)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        gr.append(grid)
        iddd.append(i)

    print("The best AUC of SVM: ", max(best_auc))
    idd = np.argmax(best_auc)
    print("Number of Selected Features is: ", iddd[idd])
    ft = features[idd] 

    # Save the model
    dump(gr[idd], "../results/svc.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv('../results/svc_features.csv')

def train_svc(X_train, y_train, X_test, y_test):
    # Create svm cross validation
    grid = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], \
            'kernel': ['rbf', 'sigmoid']}

    clf = SVC(random_state=7, probability=True)
    svc_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    svc_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    svc_pred = svc_grid.predict(X_test)
    
    roc_auc = round(roc_auc_score (y_test, svc_pred), 3)

    return (roc_auc, svc_grid)
    
