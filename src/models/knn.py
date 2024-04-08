from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train, X_test, y_test):

    # Create knn cross validation
    grid = {'n_neighbors': list(range(1, 31))}
    clf = KNeighborsClassifier()
    knn_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    knn_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    knn_pred = knn_grid.predict(X_test)
    roc_auc = round(roc_auc_score (y_test, knn_pred), 3)

    return (roc_auc, knn_grid)