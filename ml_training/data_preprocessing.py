import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define directories
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'data/models'
SCALERS_DIR = os.path.join(MODELS_DIR, 'scalers')

# Create directories if not exist
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

def preprocess_diabetes():
    # Load dataset and set column names
    df = pd.read_csv(
        os.path.join(DATA_RAW_DIR, 'diabetes.csv'),
        header=None
    )
    df.columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                  'insulin', 'bmi', 'diabetes_pedigree_function', 'age', 'outcome']

    X = df.drop('outcome', axis=1)
    y = df['outcome']

    # Handle missing values if any (assuming 0 means missing for some columns)
    cols_with_zero_as_nan = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    for col in cols_with_zero_as_nan:
        X[col] = X[col].replace(0, np.nan)
    X.fillna(X.median(), inplace=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save preprocessed data and scaler
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_train_diabetes.npy'), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_test_diabetes.npy'), X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_train_diabetes.npy'), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_test_diabetes.npy'), y_test)
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'diabetes_scaler.joblib'))

    print("Diabetes preprocessing complete.")

def preprocess_heart():
    df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'heart.csv'))

    # If needed, rename or fix missing headers
    # Assuming target column is named 'target'
    # Ensure data cleanliness (fill missing, convert categorical)
    if 'target' not in df.columns:
        raise ValueError("Expected target column 'target' missing in heart.csv")

    X = df.drop('target', axis=1)
    y = df['target']

    # Fill missing values if any
    X.fillna(X.median(), inplace=True)

    # Encode categorical columns if any
    categorical_cols = []
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_train_heart.npy'), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_test_heart.npy'), X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_train_heart.npy'), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_test_heart.npy'), y_test)
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'heart_scaler.joblib'))

    print("Heart disease preprocessing complete.")


def preprocess_alzheimer():
    df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'alzheimer.csv'))

    drop_cols = ['PatientID', 'Ethnicity', 'DoctorInCharge']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.replace('XXXConfid', pd.NA, inplace=True)

    # Fill missing numeric values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Fill missing categorical values with mode
    for col in ['Gender', 'EducationLevel']:
        if col in df.columns:
            df[col].fillna(df[col].mode(), inplace=True)

    # Map Gender to numeric
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    df['EducationLevel'] = df['EducationLevel'].astype(int)

    target_col = 'Diagnosis'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(int)

    print(f"Number of samples before scaling: {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_train_alzheimer.npy'), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_test_alzheimer.npy'), X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_train_alzheimer.npy'), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_test_alzheimer.npy'), y_test)

    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'alzheimer_scaler.joblib'))
    print("Alzheimer preprocessing complete.")




if __name__ == '__main__':
    preprocess_diabetes()
    preprocess_heart()
    preprocess_alzheimer()
    print("All preprocessing completed successfully.")
