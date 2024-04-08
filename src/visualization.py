import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import torch

# Draw ROC curve 
def draw_roc_curve (model, X_test, y_test, method, output):  
    
    prob= model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob[:, 1], pos_label=1)
    roc_auc = round(roc_auc_score(y_test, y_pred, average=None), 3)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, linestyle='solid', label= method + ", AUC = " + str(roc_auc))
    # Title
    plt.title('ROC curve', weight='bold')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(output + '/ROC',dpi=300)
    plt.show()


# Draw ROC curve NN
def draw_roc_curve_nn (model, test_loader, method, output):  
    # Load and fit model
    model.eval()
    y_pred = []
    y_obs = []
    y_probs = []

    for idx, (x,y) in enumerate(test_loader):
        with torch.no_grad():
            x,y = x.float(), y.type(torch.LongTensor)
            pred = model(x)
            probs = torch.nn.functional.softmax(pred, dim=1)
            y_probs.extend(probs[:, 1].cpu().numpy())
            preds_class = torch.argmax(pred)
            y_pred.append(preds_class.numpy())
            y_obs.append(y.numpy()[0])
    
    # Find AUC score
    fpr, tpr, _ = roc_curve(y_obs, y_probs, pos_label=1)
    roc_auc = round(roc_auc_score(y_obs, y_pred, average=None), 3)

    # Plot smooth ROC curve
    plt.plot(fpr, tpr, linestyle='solid', label= method + ", AUC = " + str(roc_auc))
    
    # Title
    plt.title('ROC curve', weight='bold')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(output + '/ROC',dpi=300)
    plt.show()