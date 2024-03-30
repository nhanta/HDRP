# Import packages
import pandas as pd
from joblib import load
from data_loader import load_data
from metrics import eval
from visualization import draw_roc_curve
import argparse

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
    print("|         -----               FEATURE SELECTIONS             -----           |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")
    
    # Read genotype-phenotype data
    X_train, y_train, X_test, y_test, feature_names, _ = load_data(args.input)
    

    print("Logistic Regression")
    # Load the Logistic Regression Model
    lg_model = load(args.output + "/lg.joblib")
    eval(lg_model, X_test, y_test).to_csv(args.output + "/lg_evaluations.csv")
    draw_roc_curve (lg_model, X_test, y_test, args.output)
    
    print("")
 
    print("Decision-Tree RFE")
    indice_features = pd.read_csv(args.output + "/dt_features.csv")['features']
    dt_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': dt_features}).to_csv(args.output + "/dt_feature_names.csv")
    # Load and fit model
    dt_model = load(args.output + "/dt.joblib")

    # Get data with the selected features
    X_train_dt = X_train[:, indice_features]
    X_test_dt = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_dt.shape[1])

    eval(dt_model, X_test_dt, y_test).to_csv(args.output + "/dt_evaluations.csv")
    draw_roc_curve (dt_model, X_test_dt, y_test, args.output)

    print("")
    
    print("SVC RFE")
    indice_features = pd.read_csv(args.output+ "/svc_features.csv")['features']
    svc_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': svc_features}).to_csv(args.output + "/svc_feature_names.csv")
    # Load and fit model
    svc_model = load(args.output + "/svc.joblib")

    # Get data with the selected features
    X_train_svc = X_train[:, indice_features]
    X_test_svc = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_svc.shape[1])

    eval(svc_model, X_test_svc, y_test).to_csv(args.output + "/svc_evaluations.csv")
    draw_roc_curve (svc_model, X_test_svc, y_test, args.output)
