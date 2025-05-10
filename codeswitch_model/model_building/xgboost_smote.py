import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           average_precision_score, precision_recall_curve, roc_curve,
                           confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data():
    print("Loading and preparing data...")
    
    # Load the data
    textgrid_features = pd.read_csv('feature/all_textgrid_features.csv')
    opensmile_features = pd.read_csv('feature/88_features_opensmile.csv')
    
    # Get features from textgrid_features (excluding metadata and target)
    textgrid_features_only = textgrid_features.drop(['source_file', 'iu_text', 'iu_start', 'iu_end', 'cs_non_english', 'spn'], axis=1)
    print("\nDropped 'spn' feature from the dataset")
    
    # Get features from opensmile_features (excluding metadata)
    opensmile_features_only = opensmile_features.select_dtypes(include=[np.number])
    
    # Combine features
    features = pd.concat([textgrid_features_only, opensmile_features_only], axis=1)
    
    # Get target variable
    y = textgrid_features['cs_non_english']
    
    # Print information about missing values
    missing_values = features.isnull().sum()
    print("\nMissing values per feature:")
    print(missing_values[missing_values > 0])
    print(f"\nrows with missing values: {features[features.isnull().any(axis=1)]}")

    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\nOriginal training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, features.columns

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_importance, save_path=None):
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance.head(20)
    sns.barplot(x='importance', y='feature', data=top_20)
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Importance Score')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train_and_evaluate_model(best_params, X_train, X_test, y_train, y_test, feature_names):
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_evaluation_{timestamp}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    
    # Train LightGBM model
    print("\nTraining LightGBM model...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
    lgb_model.fit(X_train, y_train)
    
    # Evaluate both models
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Average Precision': average_precision_score(y_test, y_pred_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        
        # Print evaluation metrics
        print(f"\n{model_name} Model Evaluation:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance analysis
        if model_name == 'XGBoost':
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:  # LightGBM
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"\n{model_name} Top 20 Most Important Features:")
        print(feature_importance.head(20))
        
        # Save metrics to file
        with open(f"{output_dir}/{model_name.lower()}_metrics.txt", 'w') as f:
            f.write(f"{model_name} Model Evaluation Metrics:\n\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write(f"\n{model_name} Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
        
        # Generate and save plots
        plot_confusion_matrix(y_test, y_pred, f"{output_dir}/{model_name.lower()}_confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_proba, f"{output_dir}/{model_name.lower()}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_pred_proba, f"{output_dir}/{model_name.lower()}_pr_curve.png")
        plot_feature_importance(feature_importance, f"{output_dir}/{model_name.lower()}_feature_importance.png")
        
        # Save feature importance to CSV
        feature_importance.to_csv(f"{output_dir}/{model_name.lower()}_feature_importance.csv", index=False)
    
    print(f"\nAll evaluation results have been saved to the '{output_dir}' directory")
    return models, feature_importance, metrics

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Best parameters from previous optimization
    best_params = {
        'subsample': 0.7, 
        'scale_pos_weight': 17.959119496855347, 
        'reg_lambda': 1, 
        'reg_alpha': 0, 
        'n_estimators': 100, 
        'min_child_weight': 1, 
        'max_depth': 7, 
        'max_delta_step': 0, 
        'learning_rate': 0.1, 
        'gamma': 0, 
        'colsample_bytree': 0.8
    }
    
    # Train and evaluate models with best parameters
    models, feature_importance, metrics = train_and_evaluate_model(
        best_params, X_train, X_test, y_train, y_test, feature_names
    ) 