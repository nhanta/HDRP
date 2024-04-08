import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from numpy.random import seed
from numpy.random import randint
from joblib import dump
import math

# Train dicision tree models
def sfe_dt(X_train, X_test, y_train, y_test, output):
    UR = 0.3
    UR_Max = 0.3
    UR_Min = 0.001
    Max_FEs = 777
    Max_Run = 17
    Run = 1
    Cost = np.zeros([Max_FEs, Max_Run])
    grid = []
    index = []
    
    while (Run <= Max_Run):
        EFs = 1
        mask = np.random.randint(0, 2, np.size(X_train, 1))   # Initialize an Individual X
        roc_auc, dt_grid, id = fit_dt(X_train, X_test, y_train, y_test, mask)   # Calculate the Fitness of X
        Nvar = np.size(X_train, 1)                         # Number of Features in Dataset

        while (EFs <= Max_FEs):

            new_mask = np.copy(mask)
            # Non-selection operation:
            
            U_Index = np.where(mask == 1)                      # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
            UN = math.ceil(UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]                          # K=index(U)
            new_mask[K] = 0                                 # Set new_mas (K)=0 


        # Selection operation:
            if np.sum(new_mask) == 0:
                S_Index = np.where(mask == 0)               # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_mask = np.copy(mask)
                new_mask[K] = 1                             # Set new_mask (K)=1

            new_roc_auc,new_dt_grid, new_id = fit_dt(X_train, X_test, y_train, y_test, new_mask)             # Calculate the Fitness of X_New

            if new_roc_auc > roc_auc:
                mask = np.copy(new_mask)
                roc_auc = new_roc_auc
                dt_grid = new_dt_grid
                id = new_id 

            UR = (UR_Max-UR_Min)*((Max_FEs-EFs)/Max_FEs)+UR_Min  # Eq(3)
            Cost[EFs-1,Run-1] = roc_auc 
            print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, roc_auc , np.sum(mask), Run))
            EFs = EFs+1
            
        grid.append(dt_grid)
        index.append(id)
        Run = Run + 1
        
    best_auc = Cost[-1, :]

    print("The best AUC of SFE: ", max(best_auc))    
    idd = np.argmax(best_auc)
    ft = index[idd]
    gr = grid[idd]

    print("Number of Selected Features is: ", len(ft))
    
    # Save the model
    dump(gr, output + "/sfe_dt.joblib")
    pd.DataFrame({'features':ft}).to_csv(output + "/sfe_dt_features.csv")
    return Cost

# Fit the decision tree 
def fit_dt(X_train, X_test, y_train, y_test, mask):

    if len(mask) == 0:
        seed(1)
        mask = randint(1, np.size(X_train, 1))
    id = [] # Store index of features
    
    for i in range(0, np.size(X_train, 1)):
        if(mask[i]==1):
            id.append(i) 

    X = X_train[:, id]
    Target = X_test[:, id]
     
    grid = {'criterion': ["gini", "entropy"], 
            'splitter': ["best", "random"],
            'max_features': ["sqrt","log2"],
            'max_depth' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}

    clf = tree.DecisionTreeClassifier(random_state=7)
    dt_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)

    # Train the regressor
    dt_grid.fit(X, y_train)
    # Make predictions using the optimised parameters
    dt_pred = dt_grid.predict(Target)
    roc_auc = round(roc_auc_score (y_test, dt_pred), 3)
     
    return roc_auc, dt_grid, id

# Train SVM SFE models
def sfe_svm(X_train, X_test, y_train, y_test, output):
    UR = 0.3
    UR_Max = 0.3
    UR_Min = 0.001
    Max_FEs = 777
    Max_Run = 17
    Run = 1
    Cost = np.zeros([Max_FEs, Max_Run])
    grid = []
    index = []
    
    while (Run <= Max_Run):
        EFs = 1
        mask = np.random.randint(0, 2, np.size(X_train, 1))   # Initialize an Individual X
        roc_auc, dt_grid, id = fit_svm(X_train, X_test, y_train, y_test, mask)   # Calculate the Fitness of X
        Nvar = np.size(X_train, 1)                         # Number of Features in Dataset

        while (EFs <= Max_FEs):

            new_mask = np.copy(mask)
            # Non-selection operation:
            
            U_Index = np.where(mask == 1)                      # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
            UN = math.ceil(UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]                          # K=index(U)
            new_mask[K] = 0                                 # Set new_mas (K)=0 


        # Selection operation:
            if np.sum(new_mask) == 0:
                S_Index = np.where(mask == 0)               # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_mask = np.copy(mask)
                new_mask[K] = 1                             # Set new_mask (K)=1

            new_roc_auc,new_dt_grid, new_id = fit_svm(X_train, X_test, y_train, y_test, new_mask)             # Calculate the Fitness of X_New

            if new_roc_auc > roc_auc:
                mask = np.copy(new_mask)
                roc_auc = new_roc_auc
                dt_grid = new_dt_grid
                id = new_id 

            UR = (UR_Max-UR_Min)*((Max_FEs-EFs)/Max_FEs)+UR_Min  # Eq(3)
            Cost[EFs-1,Run-1] = roc_auc 
            print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, roc_auc , np.sum(mask), Run))
            EFs = EFs+1
            
        grid.append(dt_grid)
        index.append(id)
        Run = Run + 1
        
    best_auc = Cost[-1, :]

    print("The best AUC of SFE: ", max(best_auc))    
    idd = np.argmax(best_auc)
    ft = index[idd]
    gr = grid[idd]

    print("Number of Selected Features is: ", len(ft))
    
    # Save the model
    dump(gr, output + "/sfe_svm.joblib")
    pd.DataFrame({'features':ft}).to_csv(output + "/sfe_svm_features.csv")
    return Cost

# Fit SVM sfe tree model
def fit_svm(X_train, X_test, y_train, y_test, mask):

    if len(mask) == 0:
        seed(1)
        mask = randint(1, np.size(X_train, 1))
    id = [] # Store index of features
    
    for i in range(0, np.size(X_train, 1)):
        if(mask[i]==1):
            id.append(i) 

    X = X_train[:, id]
    Target = X_test[:, id]
     
    # Create svm cross validation
    grid = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], \
            'kernel': ['rbf', 'sigmoid']}

    clf = SVC(random_state=7, probability=True)
    svc_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    svc_grid.fit(X, y_train)
    # Make predictions using the optimised parameters
    svc_pred = svc_grid.predict(Target)
    roc_auc = round(roc_auc_score (y_test, svc_pred), 3)
     
    return roc_auc, svc_grid, id

# Train KNN SFE models
def sfe_knn(X_train, X_test, y_train, y_test, output):
    UR = 0.3
    UR_Max = 0.3
    UR_Min = 0.001
    Max_FEs = 777
    Max_Run = 17
    Run = 1
    Cost = np.zeros([Max_FEs, Max_Run])
    grid = []
    index = []
    
    while (Run <= Max_Run):
        EFs = 1
        mask = np.random.randint(0, 2, np.size(X_train, 1))   # Initialize an Individual X
        roc_auc, dt_grid, id = fit_knn(X_train, X_test, y_train, y_test, mask)   # Calculate the Fitness of X
        Nvar = np.size(X_train, 1)                         # Number of Features in Dataset

        while (EFs <= Max_FEs):

            new_mask = np.copy(mask)
            # Non-selection operation:
            
            U_Index = np.where(mask == 1)                      # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
            UN = math.ceil(UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]                          # K=index(U)
            new_mask[K] = 0                                 # Set new_mas (K)=0 


        # Selection operation:
            if np.sum(new_mask) == 0:
                S_Index = np.where(mask == 0)               # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_mask = np.copy(mask)
                new_mask[K] = 1                             # Set new_mask (K)=1

            new_roc_auc,new_dt_grid, new_id = fit_knn(X_train, X_test, y_train, y_test, new_mask)             # Calculate the Fitness of X_New

            if new_roc_auc > roc_auc:
                mask = np.copy(new_mask)
                roc_auc = new_roc_auc
                dt_grid = new_dt_grid
                id = new_id 

            UR = (UR_Max-UR_Min)*((Max_FEs-EFs)/Max_FEs)+UR_Min  # Eq(3)
            Cost[EFs-1,Run-1] = roc_auc 
            print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, roc_auc , np.sum(mask), Run))
            EFs = EFs+1
            
        grid.append(dt_grid)
        index.append(id)
        Run = Run + 1
        
    best_auc = Cost[-1, :]

    print("The best AUC of SFE: ", max(best_auc))    
    idd = np.argmax(best_auc)
    ft = index[idd]
    gr = grid[idd]

    print("Number of Selected Features is: ", len(ft))
    
    # Save the model
    dump(gr, output + "/sfe_knn.joblib")
    pd.DataFrame({'features':ft}).to_csv(output + "/sfe_knn_features.csv")
    return Cost

# Fit KNN sfe tree model
def fit_knn(X_train, X_test, y_train, y_test, mask):

    if len(mask) == 0:
        seed(1)
        mask = randint(1, np.size(X_train, 1))
    id = [] # Store index of features
    
    for i in range(0, np.size(X_train, 1)):
        if(mask[i]==1):
            id.append(i) 

    X = X_train[:, id]
    Target = X_test[:, id]

    # Create svm cross validation
    grid = {'n_neighbors': list(range(1, 31))}
    clf = KNeighborsClassifier()
    knn_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    knn_grid.fit(X, y_train)
    # Make predictions using the optimised parameters
    svc_pred = knn_grid.predict(Target)
    roc_auc = round(roc_auc_score (y_test, svc_pred), 3)
     
    return roc_auc, knn_grid, id