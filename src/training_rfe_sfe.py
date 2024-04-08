# Import packages
from data_loader import load_data
import pandas as pd
import models.lr as lr
import models.dt as dt
import models.svm as svm
import models.sfe as sfe
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument('input', metavar='i', type=str, help='Input path')
parser.add_argument('output', metavar='o', type=str, help='Outphut path')
args = parser.parse_args()

if __name__ == '__main__':

    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----      MACHINE LEARNING FEATURE SELECTIONS     -----           |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")

    
    # Load genotype-phenotype data 
    X_train, y_train, X_test, y_test, feature_names, _ = load_data(args.input)
    '''
    print("Logistic regression")
    start_time = time.time()
    lr.train_lr (X_train, X_test, y_train, y_test, feature_names, args.output)
    end_time = time.time()
    lr_elapsed_time = end_time - start_time
    print("Elapsed time: ", lr_elapsed_time)
    print("")

    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    dt.rfe_dt(X_train, y_train, X_test, y_test, args.output)
    end_time = time.time()
    # Calculate elapsed time
    dt_elapsed_time = end_time - start_time
    print("Elapsed time: ", dt_elapsed_time)
    print("")
    
    print("SVM RFE")
    # Start timer
    start_time = time.time()
    svm.rfe_svc(X_train, y_train, X_test, y_test, args.output)
    end_time = time.time()
    # Calculate elapsed time
    svm_elapsed_time = end_time - start_time
    print("Elapsed time: ", svm_elapsed_time)
    print("")
    '''
    
    print("KNN SFE")
    start_time = time.time()
    Cost = sfe.sfe_knn(X_train, X_test, y_train, y_test, args.output)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    
    plt.plot(Cost.mean(1), '-',label= 'KNN SFE')
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel('Classification AUC'), 
    plt.title('SFE', weight='bold')
    plt.legend(loc='best')
    plt.savefig(args.output + '/SFE',dpi=300)
    plt.show()
    print("")
  
    print("DT SFE")
    start_time = time.time()
    Cost = sfe.sfe_dt(X_train, X_test, y_train, y_test, args.output)
    end_time = time.time()
    sfe_dt_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_dt_elapsed_time)
    
    plt.plot(Cost.mean(1), '-',label= 'DT SFE')
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel('Classification AUC'), 
    plt.title('SFE', weight='bold')
    plt.legend(loc='best')
    plt.savefig(args.output + '/SFE',dpi=300)
    plt.show()
    print("")
    
    print("SVM SFE")
    start_time = time.time()
    Cost = sfe.sfe_svm(X_train, X_test, y_train, y_test, args.output)
    end_time = time.time()
    sfe_svm_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_svm_elapsed_time)
    
    plt.plot(Cost.mean(1), '-', label= 'SVM SFE')
    plt.xlabel('Number of Fitness Evaluations')
    plt.ylabel('Classification AUC'), 
    plt.title('SFE', weight='bold')
    plt.legend(loc='best')
    plt.savefig(args.output + '/SFE',dpi=300)
    plt.show()
    print("")

    print("********************************** SAVING **********************************")
    '''
    pd.DataFrame({'LR': [lr_elapsed_time], 'DT':[dt_elapsed_time], 'SVM':[svm_elapsed_time], \
        'KNN_SFE':[sfe_knn_elapsed_time], 'DT_SFE':[sfe_dt_elapsed_time], \
            'SVM_SFE':[sfe_svm_elapsed_time]}).to_csv(args.output + "/Time.csv")
    '''
  
    pd.DataFrame({'KNN_SFE':[sfe_knn_elapsed_time], 'DT_SFE':[sfe_dt_elapsed_time], \
        'SVM_SFE':[sfe_svm_elapsed_time]}).to_csv(args.output + "/Time.csv")
  
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    