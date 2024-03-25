import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path):
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train = pd.read_csv(path+ '/X_train.csv').set_index('Unnamed: 0')
    y_train = pd.read_csv(path+ '/y_train.csv')['PHENOTYPE']
    X_test = pd.read_csv(path+ '/X_test.csv').set_index('Unnamed: 0')
    y_test = pd.read_csv(path+ '/y_test.csv')['PHENOTYPE']
    
    feature_names = list(X_train.columns)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, y_train, X_test, y_test, feature_names, scaler