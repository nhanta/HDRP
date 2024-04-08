# Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import models.nn as nn
import models.lr as lr
import models.dt as dt
import models.knn as knn
import models.svm as svm
import models.sfe as sfe
from data_loader import load_data
from metrics import eval, eval_nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

import argparse
import time

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument('input', metavar='i', type=str, help='Input path')
parser.add_argument('output', metavar='o', type=str, help='Output path')
args = parser.parse_args()

def get_pca (X_train, X_test, k):
  pca = PCA(n_components=k, random_state = 0)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)

  var_exp = pca.explained_variance_ratio_.cumsum()
  var_exp = var_exp*100
  plt.bar(range(k), var_exp);
  return (X_train_pca, X_test_pca)

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----               DISEASE PREDICTION              -----          |")
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
    indice_features = pd.read_csv(args.output+ "/sfe_knn_features.csv")['features']

    # Get data with the selected features
    X_train_reduce = X_train[:, indice_features]
    X_test_reduce = X_test[:, indice_features]
    
    # Number of principal components equals to number of selected features 
    k = X_test_reduce.shape[1]
    print("Number of Selected Features: ", k)
    X_train_pca,  X_test_pca = get_pca(X_train, X_test, k)
    
    # Define the train and test dataloader for neural networks
    train_loader = torch.utils.data.DataLoader(nn.Dataset(X_train, y_train))
    test_loader = torch.utils.data.DataLoader(nn.Dataset(X_test, y_test))
    train_loader_pca = torch.utils.data.DataLoader(nn.Dataset(X_train_pca, y_train))
    test_loader_pca = torch.utils.data.DataLoader(nn.Dataset(X_test_pca, y_test))
    train_loader_reduce = torch.utils.data.DataLoader(nn.Dataset(X_train_reduce, y_train))
    test_loader_reduce = torch.utils.data.DataLoader(nn.Dataset(X_test_reduce, y_test))
    
    # Training KNN
    print("KNN")
    # For all data
    start_time = time.time()
    knn_auc_all, knn_md_all = knn.train_knn(X_train, y_train, X_test , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_all: ', knn_auc_all)
    
    # For pca data from all data
    start_time = time.time()
    knn_auc_pca, knn_md_pca = knn.train_knn(X_train_pca, y_train, X_test_pca , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_pca: ', knn_auc_pca)
    
    # For reduce data
    start_time = time.time()
    knn_auc_reduce, knn_md_reduce = knn.train_knn(X_train_reduce, y_train, X_test_reduce , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC reduce: ', knn_auc_reduce)
    
    # Training DT
    print("DT")
    # For all data
    start_time = time.time()
    dt_auc_all, dt_md_all = dt.train_dt(X_train, y_train, X_test , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_all: ', dt_auc_all)
    
    # For pca data from all data
    start_time = time.time()
    dt_auc_pca, dt_md_pca = dt.train_dt(X_train_pca, y_train, X_test_pca , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_pca: ', dt_auc_pca)
    
    # For reduce data
    start_time = time.time()
    dt_auc_reduce, dt_md_reduce = dt.train_dt(X_train_reduce, y_train, X_test_reduce , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC reduce: ', dt_auc_reduce)
    
    # Training SVM
    print("svm")
    # For all data
    start_time = time.time()
    svm_auc_all, svm_md_all = svm.train_svc(X_train, y_train, X_test , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_all: ', svm_auc_all)
    
    # For pca data from all data
    start_time = time.time()
    svm_auc_pca, svm_md_pca = svm.train_svc(X_train_pca, y_train, X_test_pca , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC_pca: ', svm_auc_pca)
    
    # For reduce data
    start_time = time.time()
    svm_auc_reduce, svm_md_reduce = svm.train_svc(X_train_reduce, y_train, X_test_reduce , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC reduce: ', svm_auc_reduce)
    
    # Training NN
    print("NN")
    # For all data
    start_time = time.time()
    nn_df_md_all = nn.define_model(X_test.shape[1])
    nn_md_all = nn.train_model(nn_df_md_all, train_loader)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    _, _, _, _, nn_auc_all = nn.test_results(nn_md_all , test_loader)
    
    # For pca data
    start_time = time.time()
    nn_df_md_pca = nn.define_model(X_test_pca.shape[1])
    nn_md_pca = nn.train_model(nn_df_md_pca, train_loader_pca)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    _, _, _, _, nn_auc_pca = nn.test_results(nn_md_pca , test_loader_pca)
    
    # For reduce data
    start_time = time.time()
    nn_df_md_reduce = nn.define_model(X_test_reduce.shape[1])
    nn_md_reduce = nn.train_model(nn_df_md_reduce, train_loader_reduce)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    _, _, _, _, nn_auc_reduce = nn.test_results(nn_md_reduce , test_loader_reduce)
    
    # Find the best model
    auc = [knn_auc_reduce, dt_auc_reduce, svm_auc_reduce, nn_auc_reduce]
    method = ['knn', 'dt', 'svm', 'nn']
    print ("The best model is: ", method[np.argmax(auc)])
    
    print("********************************** SAVING **********************************")
    eval(knn_md_reduce, X_test_reduce, y_test).to_csv(args.output + "/knn_reduce_evaluations.csv")
    eval(dt_md_reduce, X_test_reduce, y_test).to_csv(args.output + "/dt_reduce_evaluations.csv")
    eval(svm_md_reduce, X_test_reduce, y_test).to_csv(args.output + "/svm_reduce_evaluations.csv")
    eval_nn(nn_md_reduce, test_loader_reduce).to_csv(args.output + "/nn_reduce_evaluations.csv")
    print("********************************* FINISHED *********************************")
    print("")
    