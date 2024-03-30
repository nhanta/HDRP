# Import packages
from data_loader import load_data
import models.lr as lr
import models.dt as dt
import models.svm as svm
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

    print("Logistic regression")
    start_time = time.time()
    lr.train_lr (X_train, X_test, y_train, y_test, feature_names, args.output)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    print("")

    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    dt.rfe_dt(X_train, y_train, X_test, y_test, args.output)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    
    print("SVM RFE")
    # Start timer
    start_time = time.time()
    svm.rfe_svc(X_train, y_train, X_test, y_test, args.output)
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    print("********************************** SAVING **********************************")
   
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    