"""
Module 3: Machine Learning Model Development
===============================================
Trains and evaluates three ML models:
  - Linear Regression (score prediction)
  - Decision Tree Classifier (performance level)
  - Random Forest Classifier (performance level)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


# ============================
# REGRESSION MODEL
# ============================

def train_linear_regression(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a Linear Regression model for score prediction.
    
    Returns:
        dict with model, predictions, and metrics.
    """
    print("\n[>] Training Linear Regression Model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model_name': 'Linear Regression',
        'model_type': 'regression',
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mse': mean_squared_error(y_test, y_pred_test),
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    metrics['cv_r2_mean'] = cv_scores.mean()
    metrics['cv_r2_std'] = cv_scores.std()
    
    # Feature coefficients
    if feature_names:
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False, key=abs)
        metrics['coefficients'] = coef_df
    
    print(f"   R² (Train): {metrics['r2_train']:.4f}")
    print(f"   R² (Test):  {metrics['r2_test']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   CV R² Mean: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    
    # Save model
    ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, 'linear_regression.pkl')
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")
    
    return {
        'model': model,
        'predictions': y_pred_test,
        'metrics': metrics,
        'model_path': model_path
    }


# ============================
# CLASSIFICATION MODELS
# ============================

def _evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name):
    """Common evaluation logic for classifiers."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'model_name': model_name,
        'model_type': 'classification',
        'accuracy_train': accuracy_score(y_train, y_pred_train),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(
            y_test, y_pred_test,
            target_names=['Low', 'Average', 'High'],
            output_dict=True,
            zero_division=0
        ),
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    return metrics, y_pred_test


def train_decision_tree(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a Decision Tree Classifier for performance level prediction.
    """
    print("\n[>] Training Decision Tree Classifier...")
    
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    metrics, y_pred_test = _evaluate_classifier(model, X_train, y_train, X_test, y_test, 'Decision Tree')
    
    # Feature importance
    if feature_names:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        metrics['feature_importance'] = importance_df
    
    print(f"   Accuracy (Train): {metrics['accuracy_train']:.4f}")
    print(f"   Accuracy (Test):  {metrics['accuracy_test']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    print(f"   CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    
    # Save model
    ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, 'decision_tree.pkl')
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")
    
    return {
        'model': model,
        'predictions': y_pred_test,
        'metrics': metrics,
        'model_path': model_path
    }


def train_random_forest(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a Random Forest Classifier for performance level prediction.
    """
    print("\n[>] Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    metrics, y_pred_test = _evaluate_classifier(model, X_train, y_train, X_test, y_test, 'Random Forest')
    
    # Feature importance
    if feature_names:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        metrics['feature_importance'] = importance_df
    
    print(f"   Accuracy (Train): {metrics['accuracy_train']:.4f}")
    print(f"   Accuracy (Test):  {metrics['accuracy_test']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    print(f"   CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    
    # Save model
    ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")
    
    return {
        'model': model,
        'predictions': y_pred_test,
        'metrics': metrics,
        'model_path': model_path
    }


def get_feature_importance(model_result):
    """Extract feature importance from a trained model result."""
    if 'feature_importance' in model_result['metrics']:
        return model_result['metrics']['feature_importance']
    elif 'coefficients' in model_result['metrics']:
        return model_result['metrics']['coefficients']
    return None


def predict_single_student(student_data, scaler, feature_names, models_dir=None):
    """
    Predict performance for a single student.
    
    Parameters:
        student_data (dict): Dictionary of student features.
        scaler: Fitted StandardScaler.
        feature_names (list): List of feature column names.
        models_dir (str): Path to saved models directory.
    
    Returns:
        dict with predicted score and performance level.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    # Prepare feature vector
    features = []
    for fname in feature_names:
        features.append(student_data.get(fname, 0))
    
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Load models
    lr_model = joblib.load(os.path.join(models_dir, 'linear_regression.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    
    # Predictions
    predicted_score = lr_model.predict(X_scaled)[0]
    predicted_score = np.clip(predicted_score, 0, 100)
    predicted_level = rf_model.predict(X_scaled)[0]
    
    level_map = {0: 'Low', 1: 'Average', 2: 'High'}
    
    # Probabilities
    probabilities = rf_model.predict_proba(X_scaled)[0]
    
    return {
        'predicted_score': round(predicted_score, 1),
        'predicted_level': level_map.get(predicted_level, 'Unknown'),
        'probabilities': {
            'Low': round(probabilities[0] * 100, 1),
            'Average': round(probabilities[1] * 100, 1),
            'High': round(probabilities[2] * 100, 1)
        }
    }


def train_all_models(preprocessed_data):
    """
    Train all three models using preprocessed data.
    
    Parameters:
        preprocessed_data (dict): Output from data_preprocessing.preprocess_data()
    
    Returns:
        dict with all model results.
    """
    print("=" * 60)
    print("[ML] TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    feature_names = preprocessed_data['feature_names']
    
    # Train Linear Regression (score prediction)
    lr_result = train_linear_regression(
        preprocessed_data['X_train_reg'], preprocessed_data['y_train_reg'],
        preprocessed_data['X_test_reg'], preprocessed_data['y_test_reg'],
        feature_names
    )
    
    # Train Decision Tree (performance level)
    dt_result = train_decision_tree(
        preprocessed_data['X_train_clf'], preprocessed_data['y_train_clf'],
        preprocessed_data['X_test_clf'], preprocessed_data['y_test_clf'],
        feature_names
    )
    
    # Train Random Forest (performance level)
    rf_result = train_random_forest(
        preprocessed_data['X_train_clf'], preprocessed_data['y_train_clf'],
        preprocessed_data['X_test_clf'], preprocessed_data['y_test_clf'],
        feature_names
    )
    
    # Model comparison
    print("\n" + "=" * 60)
    print("[CHART] MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<22} {'Train Score':<15} {'Test Score':<15} {'CV Score':<15}")
    print("-" * 67)
    lr_m = lr_result["metrics"]
    dt_m = dt_result["metrics"]
    rf_m = rf_result["metrics"]
    lr_train = f"R2={lr_m['r2_train']:.4f}"
    lr_test = f"R2={lr_m['r2_test']:.4f}"
    lr_cv = f"{lr_m['cv_r2_mean']:.4f}"
    dt_train = f"Acc={dt_m['accuracy_train']:.4f}"
    dt_test = f"Acc={dt_m['accuracy_test']:.4f}"
    dt_cv = f"{dt_m['cv_accuracy_mean']:.4f}"
    rf_train = f"Acc={rf_m['accuracy_train']:.4f}"
    rf_test = f"Acc={rf_m['accuracy_test']:.4f}"
    rf_cv = f"{rf_m['cv_accuracy_mean']:.4f}"
    print(f"{'Linear Regression':<22} {lr_train:<15} {lr_test:<15} {lr_cv:<15}")
    print(f"{'Decision Tree':<22} {dt_train:<15} {dt_test:<15} {dt_cv:<15}")
    print(f"{'Random Forest':<22} {rf_train:<15} {rf_test:<15} {rf_cv:<15}")
    
    print("\n[OK] ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    
    return {
        'linear_regression': lr_result,
        'decision_tree': dt_result,
        'random_forest': rf_result
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import preprocess_data
    
    preprocessed = preprocess_data()
    results = train_all_models(preprocessed)
    
    # Show feature importance from Random Forest
    print("\n[INFO] Top Feature Importances (Random Forest):")
    fi = get_feature_importance(results['random_forest'])
    if fi is not None:
        print(fi.to_string(index=False))
