# Import packages
import pandas as pd
from joblib import load
from data_loader import load_data
from metrics import eval, eval_nn
from visualization import draw_roc_curve, draw_roc_curve_nn
import torch
import argparse
import models.nn as nn

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
    draw_roc_curve (lg_model, X_test, y_test, "LR", args.output)
    
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
    draw_roc_curve (dt_model, X_test_dt, y_test, "DT RFE", args.output)

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
    draw_roc_curve (svc_model, X_test_svc, y_test, "SVM RFE", args.output)
    
    print("")
    
    print("NN RFE")
    print("Neural Networks Feature Selection")
    indice_features = pd.read_csv(args.output + "/nn_features.csv")['features']
    feat_imp = pd.read_csv(args.output + "/nn_importance.csv")['importance']
    nn_features = [feature_names[i] for i in indice_features]
    imp = [feat_imp[i] for i in indice_features]
    pd.DataFrame({'features': nn_features}).to_csv(args.output + "/nn_feature_names.csv")

    # Number of selected features
    k_features = len(nn_features)
    print("Number of Selected Features: ", k_features)
    # Prepair the data
    _, test_loader = nn.make_new_dataset(indice_features, X_train, X_test, y_train, y_test)
    
    # Load and evaluate the neural network model 
    nn_model = nn.define_model(k_features)()
    print(args.output + "/best_nn_model.pt")
    nn_model.load_state_dict(torch.load(args.output + "/best_nn_model.pt"))
    eval_nn(nn_model, test_loader).to_csv(args.output + "/nn_evaluations.csv")
    #nn.visualize_importances(nn_features, imp, args.output)
    draw_roc_curve_nn (nn_model, test_loader, 'NN RFE', args.output)
    
    print("")
    
    print("KNN SFE")
    indice_features = pd.read_csv(args.output+ "/sfe_knn_features.csv")['features']
    sfe_knn_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': sfe_knn_features}).to_csv(args.output + "/sfe_knn_feature_names.csv")
    # Load and fit model
    sfe_knn_model = load(args.output + "/sfe_knn.joblib")

    # Get data with the selected features
    X_test_sfe_knn = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_sfe_knn.shape[1])

    eval(sfe_knn_model, X_test_sfe_knn, y_test).to_csv(args.output + "/sfe_knn_evaluations.csv")
    draw_roc_curve (sfe_knn_model, X_test_sfe_knn, y_test, 'KNN SFE', args.output)
    
    print("")
  
    print("DT SFE")
    indice_features = pd.read_csv(args.output+ "/sfe_dt_features.csv")['features']
    sfe_dt_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': sfe_dt_features}).to_csv(args.output + "/sfe_dt_feature_names.csv")
    # Load and fit model
    sfe_dt_model = load(args.output + "/sfe_dt.joblib")

    # Get data with the selected features
    X_test_sfe_dt = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_sfe_dt.shape[1])

    eval(sfe_dt_model, X_test_sfe_dt, y_test).to_csv(args.output + "/sfe_dt_evaluations.csv")
    draw_roc_curve (sfe_dt_model, X_test_sfe_dt, y_test, 'DT SFE', args.output)
    
    print("")

    print("SVM SFE")
    indice_features = pd.read_csv(args.output+ "/sfe_svm_features.csv")['features']
    sfe_svm_features = [feature_names[i] for i in indice_features]
    pd.DataFrame({'features': sfe_svm_features}).to_csv(args.output + "/sfe_svm_feature_names.csv")
    # Load and fit model
    sfe_svm_model = load(args.output + "/sfe_svm.joblib")

    # Get data with the selected features
    X_test_sfe_svm = X_test[:, indice_features]
    print("Number of Selected Features: ", X_test_sfe_svm.shape[1])

    eval(sfe_svm_model, X_test_sfe_svm, y_test).to_csv(args.output + "/sfe_svm_evaluations.csv")
    draw_roc_curve (sfe_svm_model, X_test_sfe_svm, y_test, 'SVM SFE', args.output)

    
    