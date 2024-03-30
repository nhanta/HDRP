# Import packages
import pandas as pd
from joblib import load
from data_loader import load_data
from metrics import eval_nn
from visualization import draw_roc_curve
import argparse
import torch
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
    nn.visualize_importances(nn_features, imp, args.output)
    #draw_roc_curve (dt_model, X_test_dt, y_test, args.output)
    print("")
    


