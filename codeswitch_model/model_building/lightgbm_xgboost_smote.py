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
from tqdm import tqdm

def load_and_prepare_data():
    print("Loading and preparing data...")
    
    # Load the combined data
    combined_features = pd.read_csv('feature/combined_features.csv')
    print(f"\nLoaded combined features shape: {combined_features.shape}")
    
    # Get target variable before dropping
    y = combined_features['cs_non_english']
    
    # Drop metadata and target columns
    columns_to_drop = [
        'source_file',  'iu_text',  
        'iu_start', 'iu_end', 'iu_index',       
        'cs_non_english', 'spn'                  
    ]
    
    features = combined_features.drop(columns_to_drop, axis=1)
    print("\nDropped metadata columns and target variable")
    print(f"Features shape after dropping columns: {features.shape}")
    
    # Print information about missing values
    missing_values = features.isnull().sum()
    missing_count = missing_values[missing_values > 0].shape[0]
    if missing_count > 0:
        print(f"\nFound {missing_count} columns with missing values:")
        print(missing_values[missing_values > 0])
    print(f"\nRows with missing values: {features[features.isnull().any(axis=1)].shape[0]}")
    
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
    print(f"Test set shape: {X_test_scaled.shape}")
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

def optimize_ensemble(xgb_proba, lgb_proba, y_test):
    """Optimize ensemble weights and threshold using multiple metrics"""
    # Define parameter ranges
    weights = np.linspace(0.2, 0.8, 20)     # 20 points between 0.2 and 0.8
    thresholds = np.linspace(0.2, 0.8, 20)  # 20 points between 0.2 and 0.8
    
    # Initialize best parameters
    best_params = {
        'f1': {'weight': 0.5, 'threshold': 0.5, 'score': 0},
        'balanced_accuracy': {'weight': 0.5, 'threshold': 0.5, 'score': 0},
        'precision_recall_product': {'weight': 0.5, 'threshold': 0.5, 'score': 0}
    }
    
    # Store all results for analysis
    results = []
    
    print("\nOptimizing ensemble parameters...")
    for weight in tqdm(weights, desc="Searching weights"):
        for threshold in thresholds:
            # Create ensemble predictions
            ensemble_proba = weight * xgb_proba + (1 - weight) * lgb_proba
            ensemble_pred = (ensemble_proba >= threshold).astype(int)
            
            # Calculate various metrics
            f1 = f1_score(y_test, ensemble_pred)
            precision = precision_score(y_test, ensemble_pred)
            recall = recall_score(y_test, ensemble_pred)
            balanced_acc = (precision + recall) / 2
            prec_recall_product = precision * recall
            
            # Store results
            results.append({
                'weight': weight,
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'balanced_accuracy': balanced_acc,
                'precision_recall_product': prec_recall_product
            })
            
            # Update best parameters for each metric
            if f1 > best_params['f1']['score']:
                best_params['f1'] = {'weight': weight, 'threshold': threshold, 'score': f1}
            
            if balanced_acc > best_params['balanced_accuracy']['score']:
                best_params['balanced_accuracy'] = {'weight': weight, 'threshold': threshold, 'score': balanced_acc}
            
            if prec_recall_product > best_params['precision_recall_product']['score']:
                best_params['precision_recall_product'] = {'weight': weight, 'threshold': threshold, 'score': prec_recall_product}
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Print optimization results
    print("\nBest parameters found:")
    for metric, params in best_params.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"XGBoost weight: {params['weight']:.3f}")
        print(f"LightGBM weight: {1-params['weight']:.3f}")
        print(f"Threshold: {params['threshold']:.3f}")
        print(f"Score: {params['score']:.3f}")
    
    # Create heatmaps for different metrics
    metrics_to_plot = ['f1', 'precision', 'recall', 'balanced_accuracy', 'precision_recall_product']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx < len(axes):
            pivot_table = results_df.pivot_table(
                values=metric,
                index='weight',
                columns='threshold',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, ax=axes[idx], cmap='YlOrRd')
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Heatmap')
            axes[idx].set_xlabel('Threshold')
            axes[idx].set_ylabel('XGBoost Weight')
    
    plt.tight_layout()
    
    # Return the best parameters based on F1 score (you can change this to use a different metric)
    return best_params['f1']['weight'], best_params['f1']['threshold'], results_df

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
    
    # Get predictions for both models
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    
    # Optimize ensemble parameters
    best_weight, best_threshold, optimization_results = optimize_ensemble(xgb_proba, lgb_proba, y_test)
    
    # Save optimization results
    optimization_results.to_csv(f"{output_dir}/ensemble_optimization_results.csv", index=False)
    plt.savefig(f"{output_dir}/ensemble_optimization_heatmaps.png")
    plt.close()
    
    # Create final ensemble predictions
    ensemble_proba = best_weight * xgb_proba + (1 - best_weight) * lgb_proba
    ensemble_pred = (ensemble_proba >= best_threshold).astype(int)
    
    # Evaluate all models
    models = {
        'XGBoost': (xgb_model, xgb_proba),
        'LightGBM': (lgb_model, lgb_proba),
        'Ensemble': (None, ensemble_proba)
    }
    
    all_metrics = {}
    for model_name, (model, pred_proba) in models.items():
        print(f"\nEvaluating {model_name} model...")
        
        # Make predictions
        if model_name == 'Ensemble':
            y_pred = ensemble_pred
            y_pred_proba = ensemble_proba
        else:
            y_pred = (pred_proba >= 0.5).astype(int)
            y_pred_proba = pred_proba
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Average Precision': average_precision_score(y_test, y_pred_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        all_metrics[model_name] = metrics
        
        # Print evaluation metrics
        print(f"\n{model_name} Model Evaluation:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance analysis (only for individual models)
        if model_name != 'Ensemble':
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} Top 20 Most Important Features:")
            print(feature_importance.head(20))
            
            # Save feature importance to CSV
            feature_importance.to_csv(f"{output_dir}/{model_name.lower()}_feature_importance.csv", index=False)
            
            # Plot feature importance
            plot_feature_importance(feature_importance, f"{output_dir}/{model_name.lower()}_feature_importance.png")
        
        # Save metrics to file
        with open(f"{output_dir}/{model_name.lower()}_metrics.txt", 'w') as f:
            f.write(f"{model_name} Model Evaluation Metrics:\n\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write(f"\n{model_name} Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            if model_name == 'Ensemble':
                f.write(f"\nEnsemble Parameters:\n")
                f.write(f"XGBoost weight: {best_weight:.2f}\n")
                f.write(f"LightGBM weight: {1-best_weight:.2f}\n")
                f.write(f"Threshold: {best_threshold:.2f}\n")
        
        # Generate and save plots
        plot_confusion_matrix(y_test, y_pred, f"{output_dir}/{model_name.lower()}_confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_proba, f"{output_dir}/{model_name.lower()}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_pred_proba, f"{output_dir}/{model_name.lower()}_pr_curve.png")
    
    # Compare models
    comparison_df = pd.DataFrame(all_metrics).round(4)
    print("\nModel Comparison:")
    print(comparison_df)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv")
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar')
    plt.title('Model Comparison Across Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()
    
    print(f"\nAll evaluation results have been saved to the '{output_dir}' directory")
    return models, feature_importance, all_metrics

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