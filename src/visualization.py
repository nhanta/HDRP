import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay   
from sklearn.metrics import classification_report, accuracy_score, \
                            confusion_matrix, precision_score, auc, \
                            recall_score, f1_score,roc_auc_score, roc_curve, \
                            RocCurveDisplay, PrecisionRecallDisplay, \
                            precision_recall_curve, matthews_corrcoef, make_scorer    
# Draw ROC curve 
def draw_roc_curve (model, X_test, y_test):  
    pred_prob = model.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    # Roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    # Plot roc curves
    plt.plot(fpr, tpr, linestyle='--',color='orange', label='DT-RFE')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # Title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show()
