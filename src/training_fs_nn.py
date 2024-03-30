# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import models.nn as nn
from data_loader import load_data

import argparse
import time

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument('input', metavar='i', type=str, help='Input path')
parser.add_argument('output', metavar='o', type=str, help='Output path')
args = parser.parse_args()

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----        NEURAL NETWORK FEATURE SELECTION       -----          |")
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
    d = len(feature_names)
    
    # Define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(nn.Dataset(X_train, y_train))
    test_loader = torch.utils.data.DataLoader(nn.Dataset(X_test, y_test))

    print ("Start to train models")
    # Build a Baseline Model

    torch.manual_seed(1)

    model = nn.Model(d)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc = list(),list()
    feat_imp = np.zeros(d)
    num_epochs = 200
    
    start_time = time.time()
    for epoch in range(num_epochs):
        losses = 0 
        for idx, (x,y) in enumerate(train_loader):
            x,y = x.float(), y.type(torch.LongTensor)
            x.requires_grad=True
            optimizer.zero_grad()
            # Check if the programa can be run with model(x) and model.forward()
            preds=model.forward(x)
            loss=criterion(preds,y)
            x.requires_grad = False
            loss.backward()
            optimizer.step()
            losses+=loss.item()
        total_loss.append(losses/len(train_loader))
        if epoch%20==0:
            print("Epoch:", str(epoch+1), "\tLoss:", total_loss[-1])

    # Save the model
    #torch.save(model.state_dict(), args.output + "/nn_model.pt")
    #print("Save the model")

    model.eval()
    correct=0

    with torch.no_grad():
        y_pred = []
        y_obs = []

        for idx, (x,y) in enumerate(test_loader):
        
            x,y = x.float(), y.type(torch.LongTensor)
            pred = model(x)
            preds_class = torch.argmax(pred)
            y_pred.append(preds_class.numpy())
            y_obs.append(y.numpy()[0])
        
        # Find scores 
        acr = accuracy_score(y_obs, y_pred)
        f1 = f1_score (y_obs, y_pred)
        pre_score = precision_score(y_obs, y_pred)
        rc_score = recall_score (y_obs, y_pred)
        pre, recall, _ = precision_recall_curve(y_obs, y_pred)
        roc_auc = roc_auc_score (y_obs, y_pred)

    print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)

    test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    
    print("Calculate the Integrated Gradients")
    # Calculate the Integrated Gradients
    ig = IntegratedGradients(model)

    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(test_input_tensor, target = 1, return_convergence_delta  = True)
    attr = attr.detach().numpy()

    feat_imp =  np.mean(np.abs(attr), axis=0)

    best_auc = list()
    all_scores = []

    print("Find the best AUC")
    md = [] # Fitted model
    idf = [] # Indice of important features
    sorted_features =  [b for (a,b) in sorted(zip(feat_imp,feature_names), reverse = True)]
    for i in range(1, d):
        k_features=i
        print("For ", k_features, " features" )
        features = sorted_features[0:k_features]
        indice_features = [feature_names.index(x) for x in features]
        train_loader, test_loader = nn.make_new_dataset(indice_features, X_train, X_test, y_train, y_test)
        Model = nn.define_model(k_features)
        trained_model = nn.train_model(Model, train_loader)
        acr, pre_score, rc_score, f1, roc_auc = nn.test_results(trained_model,test_loader)
        idf.append(indice_features)
        best_auc.append(roc_auc)
        all_scores.append([acr, pre_score, rc_score, f1, roc_auc])
        md.append(trained_model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    print("")
    print("The best AUC:", max(best_auc))
    id = np.argmax(best_auc)
    indice_important_features = idf[id]
    results = all_scores[id]
    best_model = md[id]
    print(results)
    print("********************************** SAVING **********************************")
    torch.save(best_model.state_dict(), args.output + "/best_nn_model.pt")
    print("Save the model")
    pd.DataFrame({'features':indice_important_features}).to_csv(args.output + "/nn_features.csv")
    pd.DataFrame({'importance':feat_imp}).to_csv(args.output + "/nn_importance.csv")
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    