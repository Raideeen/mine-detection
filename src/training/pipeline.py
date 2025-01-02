from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --- Load and Clean Data ---
def load_and_clean_data(file_paths):
    # Load datasets
    dataframes = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(dataframes, ignore_index=True)

    # Drop NaN values
    data.dropna(inplace=True)

    # Remove outliers
    q1, q3 = np.percentile(data["Magnitude"], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data["Magnitude"] >= lower_bound) & (data["Magnitude"] <= upper_bound)]

    # Drop "Magnitude" feature (make the problem too easy)
    data = data.drop(["Magnitude"], axis=1)

    # Drop "Altitude"
    data = data.drop(["Altitude"], axis=1)

    # Drop "Pressure"
    data = data.drop(["Pressure"], axis=1)

    print(data)
    return data


# --- Data Augmentation ---
def augment_data(df, num_copies=5):
    augmented_data = []
    for _ in range(num_copies):
        augmented = df.copy()
        augmented["Mag_X"] += np.random.normal(0, 0.5, size=len(df))
        augmented["Mag_Y"] += np.random.normal(0, 0.5, size=len(df))
        augmented["Mag_Z"] += np.random.normal(0, 0.5, size=len(df))
        augmented_data.append(augmented)
    return pd.concat(augmented_data, ignore_index=True)


# --- Train-Test Split ---
def split_data(df):
    X = df.drop(columns=["Label", "Timestamp"])
    y = df["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    return X_train, X_test, y_train, y_test, scaler


# --- Model Training ---
def train_model(X_train, y_train):
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return model


# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Confidence scores

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred, y_proba


# --- Save Model and Scaler ---
def save_model_and_scaler(
    model, scaler, model_dir=Path(__file__).resolve().parent.parent.parent / "model"
):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


# --- Full Pipeline ---
def full_pipeline(file_paths):
    """
    Executes the full machine learning pipeline which includes the following steps:
    1. Load and clean data
    2. Augment data
    3. Split data into training and testing sets
    4. Train the model
    5. Evaluate the model
    6. Save the model and scaler

    Args:
        file_paths (list of str): List of file paths to the data files.

    Returns:
        None
    """
    # Step 1: Load and clean data
    print("Loading and cleaning data...")
    data = load_and_clean_data(file_paths)

    # Step 2: Augment data
    print("Augmenting data...")
    augmented_data = augment_data(data)

    # Step 3: Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test, scaler = split_data(augmented_data)

    # Step 4: Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Step 6: Save model and scaler
    print("Saving model and scaler...")
    save_model_and_scaler(model, scaler)

    print("Pipeline complete!")


# --- Run Pipeline ---
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    print(data_dir)
    datasets = [
        data_dir / "dataset_metal_dirt.csv",
        data_dir / "dataset_metal_no_dirt.csv",
        data_dir / "dataset_no_metal_dirt.csv",
        data_dir / "dataset_no_metal_no_dirt.csv",
    ]

    full_pipeline(datasets)
