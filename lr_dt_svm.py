# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import stratification.pca as pca
import models.lr as lr
import models.dt as dt
import models.svm as svm

if __name__ == '__main__':

    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----               FEATURE SELECTIONS             -----          |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")
    
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train_init = pd.read_csv('data/X_train.csv').set_index('sample')
    y_train = pd.read_csv('data/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    X_test_init = pd.read_csv('data/X_test.csv').set_index('sample')
    y_test = pd.read_csv('data/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    print("")
    
    # Choose 5 principle components
    k=5

    train_pca, test_pca = pca.get_pca (X_train_init.iloc[:, 0:-1], X_test_init.iloc[:, 0:-1], k)

    train_pca = pd.DataFrame(train_pca, columns = ["PC" + str(i) for i in range(k)])
    test_pca = pd.DataFrame(test_pca, columns = ["PC" + str(i) for i in range(k)])

    X_train = pd.concat([X_train_init.reset_index(drop=True), train_pca.reset_index(drop=True)], axis = 1)
    X_test = pd.concat([X_test_init.reset_index(drop=True), test_pca.reset_index(drop=True)], axis = 1)
    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("Logistic regression")
    acr, pre_score, rc_score, f1, roc_auc = lr.train_lr (X_train, X_test, y_train, y_test, feature_names)
    print("")
    print("Decision-Tree RFE")
    dt.rfe_dt(X_train, y_train, X_test, y_test)
    print("")
    print("SVM RFE")
    svm.rfe_svm(X_train, y_train, X_test, y_test)

    print("********************************** SAVING **********************************")
   
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    