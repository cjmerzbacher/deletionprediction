"""
Train and evaluate HistGradientBoosting models for CHO cell essentiality classification.
Uses k-fold cross validation and hyperparameter optimization.

"""

print('Beginning script...')

# Standard imports
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
import pandas as pd
import random
import numpy as np
from xgboost import XGBClassifier
import argparse

# Constants
LOCAL_DATA_FOLDER = './data'
RANDOM_SEED = 42

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model configuration."""
    parser = argparse.ArgumentParser(
        prog='Trainer',
        description='Trains a HistGradBoost model on CHO data'
    )
    parser.add_argument('--repeats', type=int, default=1, help='Number of training repeats')
    parser.add_argument('--savepath', type=str, required=True, help='Path to save model and results')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor')
    parser.add_argument('--max_depth', type=float, default=None, help='Maximum depth of trees')
    parser.add_argument('--max_iter', type=int, required=True, help='Maximum number of iterations')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    return parser.parse_args()

def load_data(local_data_folder: str) -> tuple:
    """Load and prepare CHO knockout data."""
    print('Loading data...')
    essential_data = np.load(f'{local_data_folder}/cho_essential_full.npz')
    nonessential_data = np.load(f'{local_data_folder}/cho_nonessential_full.npz')
    
    essential_names = np.array(list(set(essential_data['z'])))
    nonessential_names = np.array(list(set(nonessential_data['z'])))
    
    return essential_data, nonessential_data, essential_names, nonessential_names

def prepare_test_data(essential_data: dict, nonessential_data: dict, test_names: list) -> tuple:
    """Prepare test dataset from essential and nonessential data."""
    # Get test data for essential genes
    X_test_essential = essential_data['x'][np.isin(essential_data['z'], test_names)]
    y_test_essential = essential_data['y'][np.isin(essential_data['z'], test_names)]
    knockout_test_essential = essential_data['z'][np.isin(essential_data['z'], test_names)]

    # Get test data for nonessential genes
    X_test_nonessential = nonessential_data['x'][np.isin(nonessential_data['z'], test_names)]
    y_test_nonessential = nonessential_data['y'][np.isin(nonessential_data['z'], test_names)]
    knockout_test_nonessential = nonessential_data['z'][np.isin(nonessential_data['z'], test_names)]

    # Combine essential and nonessential data
    X_test = np.concatenate((X_test_essential, X_test_nonessential))
    y_test = np.concatenate((y_test_essential, y_test_nonessential))
    knockout_test = np.concatenate((knockout_test_essential, knockout_test_nonessential))

    return X_test, y_test, knockout_test

def prepare_fold_data(fold: int, val_df: pd.DataFrame, essential_data: dict, 
                     nonessential_data: dict, test_names: list) -> tuple:
    """Prepare training and validation data for a specific fold."""
    # Get validation names for this fold
    val_names = list(val_df[val_df['fold'] == fold]['knockout'])
    
    # Get non-test names
    nontest_names_essential = list(set(essential_data['z']) - set(test_names))
    nontest_names_nonessential = list(set(nonessential_data['z']) - set(test_names))
    
    # Get training names (non-test names minus validation names)
    essential_names_train = list(set(nontest_names_essential) - set(val_names))
    nonessential_names_train = list(set(nontest_names_nonessential) - set(val_names))

    # Prepare training data
    essential_data_train = essential_data['x'][np.isin(essential_data['z'], essential_names_train)]
    nonessential_data_train = nonessential_data['x'][np.isin(nonessential_data['z'], nonessential_names_train)]
    y_train_essential = essential_data['y'][np.isin(essential_data['z'], essential_names_train)]
    y_train_nonessential = nonessential_data['y'][np.isin(nonessential_data['z'], nonessential_names_train)]

    # Prepare validation data
    essential_data_val = essential_data['x'][np.isin(essential_data['z'], val_names)]
    nonessential_data_val = nonessential_data['x'][np.isin(nonessential_data['z'], val_names)]
    y_val_essential = essential_data['y'][np.isin(essential_data['z'], val_names)]
    y_val_nonessential = nonessential_data['y'][np.isin(nonessential_data['z'], val_names)]
    
    # Get knockout information for validation set
    knockout_essential = essential_data['z'][np.isin(essential_data['z'], val_names)]
    knockout_nonessential = nonessential_data['z'][np.isin(nonessential_data['z'], val_names)]

    # Combine data
    X_train = np.concatenate((essential_data_train, nonessential_data_train))
    y_train = np.concatenate((y_train_essential, y_train_nonessential))
    X_val = np.concatenate((essential_data_val, nonessential_data_val))
    y_val = np.concatenate((y_val_essential, y_val_nonessential))
    knockout_val = np.concatenate((knockout_essential, knockout_nonessential))

    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train, X_val, y_val, knockout_val

def train_and_evaluate_model(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           knockout_val: np.ndarray, knockout_test: np.ndarray,
                           model_savepath: str, params: dict, fold: int = 0, i: int = 0) -> tuple:
    """Train model and evaluate on validation and test sets."""
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(
            max_depth=params['max_depth'],
            max_iter=params['max_iter'],
            learning_rate=params['learning_rate'],
            random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Save trained model
    model_filename = (f'max_depth_{params["max_depth"]}_max_iter_{params["max_iter"]}_'
                     f'learning_rate_{params["learning_rate"]}_{fold}_{i}_{args.test_split}_HGB.pkl')
    with open(f'{LOCAL_DATA_FOLDER}{model_savepath}{model_filename}', 'wb') as f:
        pickle.dump(pipeline, f)

    # Helper function to get predictions and create results DataFrame
    def get_predictions_df(X, y, knockouts):
        y_pred = pipeline.predict(X)
        y_pred_proba = pipeline.predict_proba(X)
        return pd.DataFrame({
            'true_label': y,
            'prediction': y_pred,
            'score': y_pred_proba[:, 1],
            'model': 'HGB',
            'knockout_name': knockouts
        }), accuracy_score(y, y_pred)

    # Get validation and test results
    val_results, val_accuracy = get_predictions_df(X_val, y_val, knockout_val)
    test_results, test_accuracy = get_predictions_df(X_test, y_test, knockout_test)
    
    return val_accuracy, val_results, test_accuracy, test_results

def save_results(val_results: pd.DataFrame, test_results: pd.DataFrame, 
                params: dict, fold: int) -> None:
    """Save validation and test results for a single fold."""
    for results, split in [(val_results, 'val'), (test_results, 'test')]:
        filename = (f'max_depth_{params["max_depth"]}_max_iter_{params["max_iter"]}_'
                   f'learning_rate_{params["learning_rate"]}_{args.test_split}_'
                   f'results_{split}_{fold}.csv')
        results.to_csv(f'{LOCAL_DATA_FOLDER}{args.savepath}{filename}', index=False)

def save_combined_results(val_results: pd.DataFrame, test_results: pd.DataFrame, 
                         params: dict) -> None:
    """Save combined results across all folds."""
    for results, split in [(val_results, 'val'), (test_results, 'test')]:
        filename = (f'max_depth_{params["max_depth"]}_max_iter_{params["max_iter"]}_'
                   f'learning_rate_{params["learning_rate"]}_{args.test_split}_'
                   f'all_results_{split}.csv')
        results.to_csv(f'{LOCAL_DATA_FOLDER}{args.savepath}{filename}', index=False)

def run_kfold_cv(params: dict, data: list) -> dict:
    """Run k-fold cross validation with given parameters."""
    essential_data, nonessential_data, test_names = data
    
    # Prepare test set
    print('Preparing test set...')
    test_data = prepare_test_data(essential_data, nonessential_data, test_names)
    X_test, y_test, knockout_test = test_data
    
    # Load validation splits
    val_df = pd.read_csv(f'{LOCAL_DATA_FOLDER}cho_essentiality_validation_split.csv')
    
    results = {
        'accuracy_scores': [],
        'val_results': pd.DataFrame(),
        'test_results': pd.DataFrame()
    }
    
    # Run k-fold CV
    for fold in range(5):
        print(f'Processing fold {fold}')
        fold_data = prepare_fold_data(fold, val_df, essential_data, nonessential_data, test_names)
        X_train, y_train, X_val, y_val, knockout_val = fold_data
        
        # Train and evaluate
        val_accuracy, val_results, test_accuracy, test_results = train_and_evaluate_model(
            X_train, X_val, X_test, y_train, y_val, y_test,
            knockout_val, knockout_test, args.savepath, params, fold
        )
        
        print(f'Fold {fold} - Validation Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}')
        
        # Store results
        val_results['fold'] = test_results['fold'] = fold
        results['accuracy_scores'].append(val_accuracy)
        results['val_results'] = pd.concat([results['val_results'], val_results])
        results['test_results'] = pd.concat([results['test_results'], test_results])
        
        # Save fold results
        save_results(val_results, test_results, params, fold)
    
    # Save combined results
    save_combined_results(results['val_results'], results['test_results'], params)
    
    return {'loss': -np.mean(results['accuracy_scores']), 'status': 'STATUS_OK'}

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings('ignore')

    # Load data
    essential_data, nonessential_data, essential_names, nonessential_names = load_data(LOCAL_DATA_FOLDER)

    # Run cross validation
    print(f'Running k-fold CV with parameters: max_depth={args.max_depth}, '
          f'max_iter={args.max_iter}, learning_rate={args.learning_rate}')
    
    params = {
        'max_depth': args.max_depth,
        'max_iter': args.max_iter,
        'learning_rate': args.learning_rate
    }
    
    test_data = pd.read_csv(f'{LOCAL_DATA_FOLDER}cho_essentiality_test_split.csv')
    test_names = test_data.loc[test_data.test == 1, 'knockout'].to_list()
    
    result = run_kfold_cv(params, [essential_data, nonessential_data, test_names])
    print('Final result:', result)
