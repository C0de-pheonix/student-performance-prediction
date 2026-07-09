"""
Module 1B: Data Preprocessing
===============================
Handles data cleaning, encoding, normalization, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath=None):
    """Load the student dataset from CSV."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'student_data.csv')
    
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_data_summary(df):
    """Generate a comprehensive data summary."""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numerical_stats': df.describe().to_dict(),
        'categorical_cols': list(df.select_dtypes(include='object').columns),
        'numerical_cols': list(df.select_dtypes(include=[np.number]).columns)
    }
    return summary


def handle_missing_values(df):
    """
    Handle missing values using appropriate imputation strategies.
    - Numerical columns: Median imputation
    - Categorical columns: Mode imputation
    """
    df_clean = df.copy()
    
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include='object').columns
    
    missing_before = df_clean.isnull().sum().sum()
    
    # Impute numerical columns with median
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"   Filled '{col}' missing values with median: {median_val}")
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"   Filled '{col}' missing values with mode: {mode_val}")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"[OK] Missing values: {missing_before} -> {missing_after}")
    
    return df_clean


def remove_duplicates(df):
    """Remove duplicate records from the dataset."""
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_after = len(df_clean)
    n_removed = n_before - n_after
    print(f"[OK] Duplicates removed: {n_removed} ({n_before} -> {n_after} rows)")
    return df_clean.reset_index(drop=True)


def encode_categorical(df):
    """
    Encode categorical variables.
    - Binary categories: Label Encoding
    - Multi-class categories: One-Hot Encoding (for model) or Label Encoding
    
    Returns encoded DataFrame and the encoders dictionary.
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Label encoding for binary features
    binary_cols = ['gender', 'extracurricular', 'internet_access', 'test_preparation']
    for col in binary_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            print(f"   Label Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Label encoding for ordinal features
    ordinal_cols = {
        'parent_education': ['High School', 'Bachelor', 'Master', 'PhD'],
        'family_income': ['Low', 'Medium', 'High']
    }
    for col, order in ordinal_cols.items():
        if col in df_encoded.columns:
            mapping = {val: idx for idx, val in enumerate(order)}
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
            encoders[col] = mapping
            print(f"   Ordinal Encoded '{col}': {mapping}")
    
    # Encode target variable
    if 'performance_level' in df_encoded.columns:
        perf_mapping = {'Low': 0, 'Average': 1, 'High': 2}
        df_encoded['performance_level_encoded'] = df_encoded['performance_level'].map(perf_mapping)
        encoders['performance_level'] = perf_mapping
        print(f"   Encoded 'performance_level': {perf_mapping}")
    
    print(f"[OK] Categorical encoding complete")
    return df_encoded, encoders


def normalize_features(df, feature_cols=None):
    """
    Normalize numerical features using StandardScaler.
    
    Returns scaled DataFrame and the scaler object.
    """
    if feature_cols is None:
        feature_cols = ['attendance_percentage', 'study_hours_per_week', 'previous_cgpa',
                        'internal_marks', 'participation_score', 'sleep_hours', 'age']
    
    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"[OK] Normalized features: {feature_cols}")
    return df_normalized, scaler


def prepare_features(df_encoded):
    """
    Prepare feature matrix X and target vectors y.
    
    Returns:
        X: Feature matrix
        y_score: Continuous target (final_exam_score)
        y_level: Categorical target (performance_level_encoded)
        feature_names: List of feature column names
    """
    feature_cols = [
        'age', 'attendance_percentage', 'study_hours_per_week',
        'previous_cgpa', 'internal_marks', 'participation_score',
        'sleep_hours', 'gender_encoded', 'extracurricular_encoded',
        'internet_access_encoded', 'parent_education_encoded',
        'family_income_encoded', 'test_preparation_encoded'
    ]
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[feature_cols].values
    y_score = df_encoded['final_exam_score'].values
    y_level = df_encoded['performance_level_encoded'].values
    
    print(f"[OK] Features prepared: {len(feature_cols)} features, {len(X)} samples")
    return X, y_score, y_level, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[OK] Data split: Train={len(X_train)}, Test={len(X_test)} (test_size={test_size})")
    return X_train, X_test, y_train, y_test


def preprocess_data(filepath=None):
    """
    Complete preprocessing pipeline.
    
    Returns:
        dict with all preprocessed data and objects.
    """
    print("=" * 60)
    print("[INFO] STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[>] Step 1: Loading Data")
    df_raw = load_data(filepath)
    
    # Step 2: Handle missing values
    print("\n[>] Step 2: Handling Missing Values")
    df_clean = handle_missing_values(df_raw)
    
    # Step 3: Remove duplicates
    print("\n[>] Step 3: Removing Duplicates")
    df_clean = remove_duplicates(df_clean)
    
    # Step 4: Encode categorical variables
    print("\n[>] Step 4: Encoding Categorical Variables")
    df_encoded, encoders = encode_categorical(df_clean)
    
    # Step 5: Prepare features and targets
    print("\n[>] Step 5: Preparing Features")
    X, y_score, y_level, feature_names = prepare_features(df_encoded)
    
    # Step 6: Normalize features
    print("\n[>] Step 6: Normalizing Features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[OK] Scaled {X_scaled.shape[1]} features")
    
    # Step 7: Split data
    print("\n[>] Step 7: Splitting Data")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X_scaled, y_score)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(X_scaled, y_level)
    
    print("\n" + "=" * 60)
    print("[OK] PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        'df_raw': df_raw,
        'df_clean': df_clean,
        'df_encoded': df_encoded,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': feature_names,
        # Regression data
        'X_train_reg': X_train_reg,
        'X_test_reg': X_test_reg,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        # Classification data
        'X_train_clf': X_train_clf,
        'X_test_clf': X_test_clf,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
    }


if __name__ == '__main__':
    result = preprocess_data()
    print(f"\n[CHART] Preprocessed Data Summary:")
    print(f"   Raw shape: {result['df_raw'].shape}")
    print(f"   Clean shape: {result['df_clean'].shape}")
    print(f"   Features: {result['feature_names']}")
    print(f"   Train samples: {len(result['X_train_reg'])}")
    print(f"   Test samples: {len(result['X_test_reg'])}")
