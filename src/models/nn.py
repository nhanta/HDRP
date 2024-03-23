
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.x = x
        self.y = y 
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

# Code a neural network with the nn module imported into the class
class Obesity_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(141, 280)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(280,560)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(560,2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        lin1_out = self.linear1(x)
        sigmoid1_out = self.sigmoid1(lin1_out)
        lin2_out = self.linear2(sigmoid1_out)
        sigmoid2_out = self.sigmoid2(lin2_out)
        lin3_out = self.linear3(sigmoid2_out)
        softmax_out = self.softmax(lin3_out)
        return softmax_out

# Print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        importance = plt.figure(figsize=(12,6), dpi=150)
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True, rotation=90, fontsize=4)
        plt.tick_params(axis='x', pad=15)
        plt.xlabel(axis_title)
        plt.title(title)
        importance.savefig('feature_importance.png')

# Get Top K Least Important Features and Retrain
def make_new_dataset(features_to_be_dropped, X_train_after, X_test_after, y_train, y_test):
    
    X_train_drop = X_train_after.drop(features_to_be_dropped, axis=1)
    X_test_drop = X_test_after.drop(features_to_be_dropped, axis=1)

    scaler = StandardScaler()
    X_train_drop = scaler.fit_transform(X_train_drop)
    X_test_drop = scaler.transform(X_test_drop)

    # define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(Dataset(X_train_drop, y_train))
    test_loader = torch.utils.data.DataLoader(Dataset(X_test_drop, y_test))
    
    return train_loader,test_loader

def define_model(k_features, len_features):
    torch.manual_seed(1)
    # code a neural network with the nn module imported into the class
    class Obesity_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(len_features-k_features, 280) # since features have been dropped chaneg input layer
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(280, 560)
            self.sigmoid2 = nn.Sigmoid()
            self.linear3 = nn.Linear(560, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self,x):
            lin1_out = self.linear1(x)
            sigmoid1_out = self.sigmoid1(lin1_out)
            lin2_out = self.linear2(sigmoid1_out)
            sigmoid2_out = self.sigmoid2(lin2_out)
            lin3_out = self.linear3(sigmoid2_out)
            softmax_out = self.softmax(lin3_out)
            return softmax_out
    return Obesity_Model

def train_model(Obesity_Model, train_loader):
    model = Obesity_Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc = list(),list()

    num_epochs = 500

    for epoch in range(num_epochs):
        losses = 0 
        for idx, (x,y) in enumerate(train_loader):
            x,y = x.float(), y.type(torch.LongTensor)
            x.requires_grad=True
            optimizer.zero_grad()
            # check if the progrma can be run with model(x) and model.forward()
            preds=model.forward(x)
            loss=criterion(preds,y)
            x.requires_grad = False
            loss.backward()
            optimizer.step()
            losses+=loss.item()
        total_loss.append(losses/len(train_loader))
        if epoch%50==0:
            print("Epoch:", str(epoch+1), "\tLoss:", total_loss[-1])
    return model

def test_results(model, test_loader):
    model.eval()
    y_pred = []
    y_obs = []

    for idx, (x,y) in enumerate(test_loader):
        with torch.no_grad():
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
    roc_auc = round(roc_auc_score (y_obs, y_pred), 3)
        
    print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)
    
    return (acr, pre_score, rc_score, f1, roc_auc)