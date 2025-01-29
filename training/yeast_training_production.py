"""
Train and evaluate multiple ML models for yeast production classification.
Models include HistGradientBoosting, LinearSVC, LogisticRegression and RandomForest,
with balanced and resampled variants.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import random
import argparse
from imblearn.under_sampling import RandomUnderSampler

# Setup paths
tools_path = './tools'
sys.path.append(tools_path)
import knockout_voting as ko
LOCAL_DATA_FOLDER = '../data/'
RANDOM_SEED = 42

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train models on betaxanthin production data')
    parser.add_argument('--repeats', type=int, default=1, help='Number of training repeats')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--savepath', type=str, required=True, help='Path to save model and results')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set fraction')
    return parser.parse_args()

def load_and_preprocess_data():
    """Load knockout data, remove NaNs, scale and bin production values."""
    print('Loading data...')
    raw_data = np.load(LOCAL_DATA_FOLDER + 'yeast_single_knockouts.npz')
    # Remove NaN values
    nan_mask = ~np.isnan(raw_data['prod'])
    data = {
        'x': raw_data['x'][nan_mask],
        'reg': raw_data['prod'][nan_mask],
        'z': raw_data['z'][nan_mask]
    }
    
    # Scale production values to [0,1]
    data['reg'] = (data['reg'] - data['reg'].min()) / (data['reg'].max() - data['reg'].min())
    
    # Bin into 3 classes
    threshs = [0.0, 0.4, 0.65, 1.0]
    data['y'] = np.array([0 if i < threshs[1] else 1 if i < threshs[2] else 2 for i in data['reg']])
    
    return data

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, 
                           knockout_val, knockout_test, model_savepath, model, 
                           model_name, split_percentage, fold):
    """Train model and evaluate on validation and test sets."""
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Save trained model
    with open(LOCAL_DATA_FOLDER + model_savepath + f'{model_name}_{fold}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Get predictions and probabilities
    def get_predictions(X):
        y_pred = pipeline.predict(X)
        y_pred_proba = (pipeline.predict_proba(X) if hasattr(pipeline, "predict_proba") 
                       else pipeline.decision_function(X))
        return y_pred, y_pred_proba

    # Create results dataframes
    def create_results_df(y_true, y_pred, y_pred_proba, knockout_names):
        return pd.DataFrame({
            'true_label': y_true,
            'fold': fold,
            'prediction': y_pred,
            'score0': y_pred_proba[:,0],
            'score1': y_pred_proba[:,1], 
            'score2': y_pred_proba[:,2],
            'model': model_name,
            'knockout_name': knockout_names
        })

    # Get validation and test results
    y_pred_val, y_proba_val = get_predictions(X_val)
    y_pred_test, y_proba_test = get_predictions(X_test)
    
    results_val = create_results_df(y_val, y_pred_val, y_proba_val, knockout_val)
    results_test = create_results_df(y_test, y_pred_test, y_proba_test, knockout_test)
    
    return results_val, results_test

def get_model_variants():
    """Define model variants with different configurations."""
    base_params = {'verbose': 1}
    balanced_params = {**base_params, 'class_weight': 'balanced'}
    
    models = {}
    for model_class in [HistGradientBoostingClassifier, LinearSVC, 
                       LogisticRegression, RandomForestClassifier]:
        name = model_class.__name__
        models.update({
            name: model_class(**base_params),
            f'{name}_Balanced': model_class(**balanced_params),
            f'{name}_Resampled': model_class(**base_params),
            f'{name}_Resampled_Balanced': model_class(**balanced_params)
        })
    return models

def main():
    """Main training loop."""
    args = parse_arguments()
    data = load_and_preprocess_data()
    
    # Get test set
    test_names = pd.read_csv(LOCAL_DATA_FOLDER + 'yeast_production_test_split.csv').values.flatten()
    nontest_names = list(set(data['z']) - set(test_names))
    
    test_mask = np.isin(data['z'], test_names)
    X_test = data['x'][test_mask]
    y_test = data['y'][test_mask]
    knockout_test = data['z'][test_mask]
    
    # Load validation splits
    val_df = pd.read_csv(LOCAL_DATA_FOLDER + 'yeast_production_validation_split.csv')
    
    # Train all model variants
    for model_name, model in get_model_variants().items():
        print(f'Training {model_name}...')
        
        for fold in range(5):
            print(f'Fold {fold}')
            
            # Get train/val split
            val_names = list(val_df[val_df['fold'] == fold]['knockout'])
            train_names = list(set(nontest_names) - set(val_names))
            
            # Prepare data
            train_mask = np.isin(data['z'], train_names)
            val_mask = np.isin(data['z'], val_names)
            
            X_train = data['x'][train_mask]
            y_train = data['y'][train_mask]
            X_val = data['x'][val_mask]
            y_val = data['y'][val_mask]
            knockout_val = data['z'][val_mask]
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Apply resampling if needed
            if 'Resampled' in model_name:
                rus = RandomUnderSampler(sampling_strategy='majority')
                X_train, y_train = rus.fit_resample(X_train, y_train)
            
            # Train and evaluate
            results_val, results_test = train_and_evaluate_model(
                X_train, X_val, X_test, y_train, y_val, y_test,
                knockout_val, knockout_test, args.savepath, model, 
                model_name, args.test_split, fold
            )
            
            # Save results
            for results, split in [(results_val, 'val'), (results_test, 'test')]:
                results.to_csv(
                    f'{LOCAL_DATA_FOLDER}{args.savepath}{model_name}_fold_{fold}_{split}_results.csv',
                    index=False
                )

if __name__ == '__main__':
    main()
