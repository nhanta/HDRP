import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path):
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train = pd.read_csv('path/X_train.csv').set_index('sample')
    y_train = pd.read_csv('path/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    X_test = pd.read_csv('path/X_test.csv').set_index('sample')
    y_test = pd.read_csv('path/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    
    '''
    # Choose 5 principle components
    k=5

    train_pca, test_pca = pca.get_pca (X_train_init.iloc[:, 0:-1], X_test_init.iloc[:, 0:-1], k)

    train_pca = pd.DataFrame(train_pca, columns = ["PC" + str(i) for i in range(k)])
    test_pca = pd.DataFrame(test_pca, columns = ["PC" + str(i) for i in range(k)])

    X_train = pd.concat([X_train_init.reset_index(drop=True), train_pca.reset_index(drop=True)], axis = 1)
    X_test = pd.concat([X_test_init.reset_index(drop=True), test_pca.reset_index(drop=True)], axis = 1)
    '''
    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, y_train, X_test, y_test, feature_names, scaler