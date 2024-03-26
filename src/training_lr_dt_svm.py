# Import packages
from data_loader import load_data
import models.lr as lr
import models.dt as dt
import models.svm as svm

import time

if __name__ == '__main__':

    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----               FEATURE SELECTIONS             -----           |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")

    
    # Load genotype-phenotype data 
    X_train, y_train, X_test, y_test, feature_names, _ = load_data('../data/obesity')
    '''
    print("Logistic regression")
    start_time = time.time()
    lr.train_lr (X_train, X_test, y_train, y_test, feature_names)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    print("")
    '''
    '''
    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    dt.rfe_dt(X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    '''
    
    print("SVM RFE")
    # Start timer
    start_time = time.time()
    svm.rfe_svc(X_train, y_train, X_test, y_test)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    

    
    #print("")
    #print("SVM RFE")
    #svm.rfe_svm(X_train, y_train, X_test, y_test)

    print("********************************** SAVING **********************************")
   
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    