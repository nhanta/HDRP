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
    X_train, y_train, X_test, y_test, feature_names, _ = load_data('../data/obesity')
    d = len(feature_names)
    X_train_ft = pd.DataFrame(data = X_train, columns = feature_names)
    X_test_ft = pd.DataFrame(data = X_test, columns = feature_names)
    
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
    num_epochs = 10

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
    torch.save(model.state_dict(), '../results/nn_model.pt')
    print("Save the model")

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


    nn.visualize_importances(feature_names, np.mean(np.abs(attr), axis=0))

    feat_imp =  np.mean(np.abs(attr), axis=0)
    [(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]

    best_accuracy = list()
    all_scores = []

    print("Find the best accuracy")
    for i in range(0, d):
        k_features=i
        print("For ",k_features, " features" )
        features_to_be_dropped = [b for (a,b) in sorted(zip(feat_imp,feature_names))][0:k_features]
        train_loader, test_loader = nn.make_new_dataset(features_to_be_dropped, X_train_ft, X_test_ft, y_train, y_test)
        Model = nn.define_model(k_features, d)
        trained_model = nn.train_model(Model, train_loader)
        acr, pre_score, rc_score, f1, roc_auc = nn.test_results(trained_model,test_loader)
        best_accuracy.append(acr)
        all_scores.append([acr, pre_score, rc_score, f1, roc_auc])

    print("The best accuracy:", max(best_accuracy))
    id = np.argmax(best_accuracy)
    results = all_scores[id]
    print(results)

    print (id)

    print('Significant features')
    ft = [(a,b) for (a,b) in sorted(zip(feat_imp,feature_names))]
    print (ft)

    print("********************************** SAVING **********************************")
    pd.DataFrame({'IFeatures':ft}).to_csv('../results/nn_IFeat.csv')
    pd.DataFrame({'eval':best_accuracy}).to_csv('../results/nn_evaluation.csv')
    pd.DataFrame({'all_score':all_scores}).to_csv('../results/nn_all_scores.csv')

    print("")
    print("********************************* FINISHED *********************************")
    print("")
    