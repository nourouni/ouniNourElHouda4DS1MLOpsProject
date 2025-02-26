import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Data Preparation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(train_path, test_path, scaler=None):
    """
    If train_path is provided, prepare both training and test data and fit/transform the scaler.
    If train_path is None, only process the test data using the provided scaler.
    """
    if train_path is not None:
        # Read both training and test data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        print("Initial training data types:")
        print(df_train.dtypes)  # Check initial column types
        
        # One-hot encoding for both datasets
        df_train = pd.get_dummies(df_train, drop_first=True)
        df_test = pd.get_dummies(df_test, drop_first=True)
        
        # Align test data columns with training data
        df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
        
        # Get numeric columns
        num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale the data
        if scaler is None:
            scaler = StandardScaler()
            df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
        else:
            df_train[num_cols] = scaler.transform(df_train[num_cols])
        df_test[num_cols] = scaler.transform(df_test[num_cols])
        
        print("Training data after scaling:")
        print(df_train.head())
        
        return df_train, df_test, scaler
    else:
        # Only test data is provided
        df_test = pd.read_csv(test_path)
        df_test = pd.get_dummies(df_test, drop_first=True)
	print("hello")
        if scaler is not None:
            # We assume that the test data already has the same columns as used during training.
            num_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            df_test[num_cols] = scaler.transform(df_test[num_cols])
        return None, df_test, scaler


def perform_grid_search(estimator, param_grid, X_train, y_train, scoring='accuracy', cv=5, verbose=1, n_jobs=-1):

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search


# Model Training
def train_model(X_train, y_train):
    base_model = DecisionTreeClassifier(max_depth=2, class_weight="balanced")
    model = AdaBoostClassifier(base_model, n_estimators=200, learning_rate=0.05, random_state=42)
    
   

    # Apply undersampling

    param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    }

    best_adaboost_model, best_params, grid_search = perform_grid_search(
    model, param_grid, X_train, y_train
    )
    model = best_adaboost_model
    
    mlflow.log_params(best_params)

    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and log metrics to MLflow.
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    
    mlflow.log_artifact("classification_report.txt")

   
    mlflow.log_artifact("confusion_matrix.csv")

    return acc, auc, report, matrix

# Save Model
def save_model(model, scaler, filename="model.pkl"):
    joblib.dump((model, scaler), filename)
    
    mlflow.log_artifact(filename)

# Load Model
def load_model(filename="model.pkl"):
    print(f"Loading model from {filename}...")
    return joblib.load(filename)
    os.makedirs("artifacts", exist_ok=True)

    # Save classification report to artifacts directory
    report_path = os.path.join("artifacts", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save confusion matrix to artifacts directory
    matrix_path = os.path.join("artifacts", "confusion_matrix.csv")
    np.savetxt(matrix_path, matrix, delimiter=",")
    mlflow.log_artifact(matrix_path)
