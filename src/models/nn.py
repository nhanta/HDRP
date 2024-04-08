
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score,roc_auc_score,  RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve

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
class Model(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.linear1 = nn.Linear(d, 2*d)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(2*d, 4*d)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(4*d, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lin1_out = self.linear1(x)
        sigmoid1_out = self.sigmoid1(lin1_out)
        lin2_out = self.linear2(sigmoid1_out)
        sigmoid2_out = self.sigmoid2(lin2_out)
        lin3_out = self.linear3(sigmoid2_out)
        softmax_out = self.softmax(lin3_out)
        return softmax_out

# Print importances and visualize distribution
def visualize_importances(feature_names, importances, output,title="Average Feature Importances", plot=True, axis_title="Features"):
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
        importance.savefig(output + "/feature_importance.png")

# Get Top K Least Important Features and Retrain
def make_new_dataset(indice_features, X_train, X_test, y_train, y_test):
    X_train_reduce = X_train[:, indice_features]
    X_test_reduce = X_test[:, indice_features]

    scaler = StandardScaler()
    X_train_reduce = scaler.fit_transform(X_train_reduce)
    X_test_reduce = scaler.transform(X_test_reduce)

    # Define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(Dataset(X_train_reduce, y_train))
    test_loader = torch.utils.data.DataLoader(Dataset(X_test_reduce, y_test))
    
    return train_loader, test_loader

def define_model(k_features):
    torch.manual_seed(1)
    # Code a neural network with the nn module imported into the class
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            d = k_features
            self.linear1 = nn.Linear(d, 2*d) # Since features have been dropped change input layer
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(2*d, 4*d)
            self.sigmoid2 = nn.Sigmoid()
            self.linear3 = nn.Linear(4*d, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self,x):
            lin1_out = self.linear1(x)
            sigmoid1_out = self.sigmoid1(lin1_out)
            lin2_out = self.linear2(sigmoid1_out)
            sigmoid2_out = self.sigmoid2(lin2_out)
            lin3_out = self.linear3(sigmoid2_out)
            softmax_out = self.softmax(lin3_out)
            return softmax_out
    return Model

def train_model(Model, train_loader):
    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss, total_acc = list(),list()

    num_epochs = 200

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
    #pre, recall, _ = precision_recall_curve(y_obs, y_pred)
    roc_auc = round(roc_auc_score (y_obs, y_pred), 3)
        
    print("accuracy:", acr,  "precision_score:", 
        pre_score, "recall_score:", rc_score, "f1_score:", f1, "roc_auc_score:", roc_auc)
    
    return (acr, pre_score, rc_score, f1, roc_auc)