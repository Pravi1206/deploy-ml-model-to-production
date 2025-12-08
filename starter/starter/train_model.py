# Script to train machine learning model.
import os
import sys
from io import StringIO

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# own imports 
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def _load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and clean it by removing all spaces.

    Inputs
    ------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    data : pd.DataFrame
        Loaded data as a pandas DataFrame with spaces removed from column names and values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create clean file path
    clean_file_path = os.path.join(os.path.dirname(file_path), "census_clean.csv")
    
    # Delete previous clean file if it exists
    if os.path.exists(clean_file_path):
        os.remove(clean_file_path)
    
    # Read the raw file content and remove all spaces
    print(f"INFO: Cleaning data from {file_path}...")
    with open(file_path, 'r') as f:
        raw_content = f.read()
    
    cleaned_content = raw_content.replace(' ', '')
    
    # Write cleaned content to a temporary file or use StringIO
    data: pd.DataFrame = pd.read_csv(StringIO(cleaned_content))
    
    # Drop rows with missing values
    print("INFO: Dropping rows with missing values...")
    data = data.dropna()
    
    # Save cleaned data to new file
    print(f"INFO: Saving cleaned data to {clean_file_path}...")
    data.to_csv(clean_file_path, index=False)
    
    return data

def _save_model(model, encoder, lb, model_path: str) -> None:
    """
    Save the trained model and preprocessing artifacts.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Fitted OneHotEncoder for categorical features.
    lb : LabelBinarizer
        Fitted LabelBinarizer for labels.
    model_path : str
        Path to the directory where artifacts will be saved.
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the model, encoder, and label binarizer
    print("INFO: Saving model and artifacts...")
    with open(f"{model_path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_path}/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(f"{model_path}/lb.pkl", "wb") as f:
        pickle.dump(lb, f)
    
    print(f"INFO: Model and artifacts saved to {model_path}/")

def compute_slice_metrics(model, test_data, cat_features, label, encoder, lb):
    """
    Compute model performance on slices of categorical features.
    
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    test_data : pd.DataFrame
        Test dataset containing features and labels.
    cat_features : list
        List of categorical feature names.
    label : str
        Name of the label column.
    encoder : OneHotEncoder
        Fitted OneHotEncoder for categorical features.
    lb : LabelBinarizer
        Fitted LabelBinarizer for labels.
    
    Returns
    -------
    slice_metrics : list of dict
        List containing performance metrics for each slice.
    """
    slice_metrics = []
    
    for feature in cat_features:
        # Get unique values for this categorical feature
        unique_values = test_data[feature].unique()
        
        for value in unique_values:
            # Filter data for this slice
            slice_data = test_data[test_data[feature] == value]
            
            if len(slice_data) == 0:
                continue
            
            # Process the slice data
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )
            
            # Make predictions
            preds = inference(model, X_slice)
            
            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            
            # Store results
            slice_metrics.append({
                'feature': feature,
                'value': value,
                'count': len(slice_data),
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta
            })
    
    return slice_metrics

def main():
    """
    Main function for training the machine learning model.
    """

    # variables
    script_dir: str    = os.path.dirname(os.path.abspath(__file__))
    parent_dir: str    = os.path.dirname(script_dir)
    data_path: str     = os.path.join(parent_dir, "data", "census.csv")
    model_path: str    = os.path.join(parent_dir, "model")
    cat_features: list = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]
    hyperparameters: dict = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    }

    try:
        # load in the data.
        print("INFO: Load and clean data...")
        data: pd.DataFrame = _load_data(data_path)

        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        print("INFO: Splitting data into train and test sets...")
        train, test = train_test_split(data, test_size=0.20)

        # process the data using the process_data function.
        print("INFO: Processing data...")
        X_train, y_train, encoder, lb = process_data(
            train, 
            categorical_features=cat_features, 
            label="salary", 
            training=True
        )

        # Proces the test data with the process_data function.
        print("INFO: Processing test data...")
        X_test, y_test, _, _ = process_data(
            test, 
            categorical_features=cat_features, 
            label="salary", 
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Train and save a model.
        print("INFO: Training model...")
        model = train_model(X_train, y_train, hyperparameters)
        
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        print(f"INFO: Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}")

        # Compute performance on slices
        print("INFO: Computing performance on data slices...")
        slice_metrics = compute_slice_metrics(model, test, cat_features, "salary", encoder, lb)
        
        # Save slice metrics to file
        slice_output_path = os.path.join(parent_dir, "model", "slice_output.txt")
        with open(slice_output_path, 'w') as f:
            f.write("Model Performance on Data Slices\n")
            f.write("=" * 80 + "\n\n")
            for metric in slice_metrics:
                f.write(f"Feature: {metric['feature']}\n")
                f.write(f"Value: {metric['value']}\n")
                f.write(f"Count: {metric['count']}\n")
                f.write(f"Precision: {metric['precision']:.4f}\n")
                f.write(f"Recall: {metric['recall']:.4f}\n")
                f.write(f"F-beta: {metric['fbeta']:.4f}\n")
                f.write("-" * 80 + "\n")
        print(f"INFO: Slice metrics saved to {slice_output_path}")

        # Save the model and artifacts
        _save_model(model, encoder, lb, model_path)

    except Exception as e:
        print(f"ERROR: An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())