from __future__ import print_function, division
import pandas as pd
from math import sqrt
from scipy.special import ndtri
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
                 
def eval (model, X_test, y_test):
    # Load and fit model
    pred = model.predict(X_test)
    # Find AUC score
    roc_auc = roc_auc_score (y_test, pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
    print('TN', TN, 'FP', FP, 'FN', FN, 'TP', TP)
    print("Roc_AUC: ", roc_auc)
    ss, sp, acc, mcc, ss_ci, sp_ci, acc_ci, mcc_ci = measure (TN, FP, FN, TP, 0.95)

    return pd.DataFrame({'Sensitivity': [ss, ss_ci], 'Specificity': [sp, sp_ci], \
                        'Accuracy': [acc, acc_ci], 'MCC': [mcc, mcc_ci], 'AUC': roc_auc})

def eval_nn (model, test_loader):

    # Load and fit model
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
   
    # Find AUC score
    roc_auc = roc_auc_score (y_obs, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_obs, y_pred).ravel()
    print('TN', TN, 'FP', FP, 'FN', FN, 'TP', TP)
    print("Roc_AUC: ", roc_auc)
    ss, sp, acc, mcc, ss_ci, sp_ci, acc_ci, mcc_ci = measure (TN, FP, FN, TP, 0.95)

    return pd.DataFrame({'Sensitivity': [ss, ss_ci], 'Specificity': [sp, sp_ci], \
                        'Accuracy': [acc, acc_ci], 'MCC': [mcc, mcc_ci], 'AUC': roc_auc})    

def proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """
    
    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    
    # 
    z = -ndtri((1.0-alpha)/2)
    
    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = proportion_confidence_interval(TP, TP + FN, z)
    
    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = proportion_confidence_interval(TN, TN + FP, z)
    
    # Compute MCC
    mcc = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + 10**(-16))
    mcc_confidence_interval = proportion_confidence_interval(TP*TN - FP*FN, sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + 10**(-16)), z)

    # Compute accuracy
    acc = (TP +TN)/(TP+FP+TN+FN)
    acc_confidence_interval = proportion_confidence_interval(TP +TN, TP+FP+TN+FN, z)

    return sensitivity_point_estimate, specificity_point_estimate, \
        sensitivity_confidence_interval, specificity_confidence_interval, \
        acc, acc_confidence_interval, mcc, mcc_confidence_interval 

# Get sensitivity, specificity, accuracy, and Matthews's correlation coefficient.

def measure (TP, FP, FN, TN, a):
    sensitivity_point_estimate, specificity_point_estimate, \
        sensitivity_confidence_interval, specificity_confidence_interval, \
        acc, acc_confidence_interval, mcc, mcc_confidence_interval \
        = sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=a)
 
    print("Sensitivity: %f, Specificity: %f, Accuracy: %f, MCC: %f" %(sensitivity_point_estimate, specificity_point_estimate, acc, mcc))
    print("alpha = %f CI for sensitivity:"%a, sensitivity_confidence_interval)
    print("alpha = %f CI for specificity:"%a, specificity_confidence_interval)
    print("alpha = %f CI for accuracy:"%a, acc_confidence_interval)
    print("alpha = %f CI for MCC:"%a, mcc_confidence_interval)
    print("")
    return (sensitivity_point_estimate, specificity_point_estimate, acc, mcc, \
            sensitivity_confidence_interval, specificity_confidence_interval, \
            acc_confidence_interval, mcc_confidence_interval)
