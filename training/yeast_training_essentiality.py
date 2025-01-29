"""
Train and evaluate RandomForest models for yeast essentiality classification.
Uses k-fold cross validation and hyperparameter optimization.
"""

import sys
import warnings
import pickle
from typing import Dict, Tuple, List
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Add tools path to system path
tools_path = './tools'
sys.path.append(tools_path)
from splitting_lists import split_lists_randomly
import knockout_voting as ko

# Constants
LOCAL_DATA_FOLDER = '../data/'
RANDOM_SEED = 42

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model configuration."""
    parser = argparse.ArgumentParser(
        prog='Trainer',
        description='Trains a RandomForest model on yeast essentiality data'
    )
    parser.add_argument('--repeats', type=int, default=1, help='Number of training repeats')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--savepath', type=str, required=True, help='Path to save model and results')
    parser.add_argument('--grid_search', type=bool, default=False, help='Use grid search hyperparameter optimization')
    parser.add_argument('--max_depth', type=int, default=None, help='Max depth of trees')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in forest')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples required to split node')

    return parser.parse_args()

def load_data() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and prepare yeast knockout data and split into train/test sets.
    
    Returns:
        data: Raw knockout data
        train_names: Essential and nonessential gene names for training
        test_names: Essential and nonessential gene names for testing
    """
    # Load raw data
    data = np.load(LOCAL_DATA_FOLDER + 'yeast_single_knockouts.npz')
    df = pd.DataFrame({'knockout': data['z'], 'essential': data['y']})
    
    # Get essential and nonessential genes
    nonessential = list(set(df.loc[df.essential == 1]['knockout']))
    essential = list(set(df.loc[df.essential == 0]['knockout']))
    
    # Load and apply predefined test split
    test_set = pd.read_csv(LOCAL_DATA_FOLDER + 'yeast_essentiality_test_split.csv')
    test_knockouts = test_set.loc[test_set.test == 1].knockout.unique()
    
    # Split genes into train/test
    test_names = (
        [e for e in essential if e in test_knockouts],
        [e for e in nonessential if e in test_knockouts]
    )
    train_names = (
        [e for e in essential if e not in test_knockouts],
        [e for e in nonessential if e not in test_knockouts]
    )
    
    return data, train_names, test_names

def create_pipeline(params: Dict) -> Pipeline:
    """Create sklearn pipeline with RandomForest classifier."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            min_samples_split=params['min_samples_split'],
            random_state=RANDOM_SEED))
    ])

def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      knockout_names: np.ndarray, params: Dict,
                      model_path: str, fold: int = 0) -> Tuple[pd.DataFrame, float]:
    """
    Train model and evaluate performance.
    
    Returns:
        results_df: DataFrame with predictions and metrics
        accuracy: Knockout-level accuracy score
    """
    # Train model
    pipeline = create_pipeline(params)
    pipeline.fit(X_train, y_train)
    
    # Save trained model
    model_name = f'rf_d{params["max_depth"]}_t{params["n_estimators"]}_s{params["min_samples_split"]}_{fold}.pkl'
    with open(f'{model_path}{model_name}', 'wb') as f:
        pickle.dump(pipeline, f)

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'true_label': y_test,
        'prediction': y_pred,
        'score': y_proba[:, 1],
        'model': 'RandomForest',
        'knockout_name': knockout_names,
        'fold': fold
    })
    
    # Calculate knockout-level accuracy
    accuracy = ko.calculate_knockout_accuracy(
        results_df, thresh=0.5, score='score', column='knockout_name'
    )
    
    return results_df, accuracy

def main():
    """Main execution function."""
    args = parse_arguments()
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings('ignore')
    
    # Load data
    data, train_names, test_names = load_data()
    
    # Set model parameters
    params = {
        'max_depth': args.max_depth or None,
        'n_estimators': args.n_estimators,
        'min_samples_split': args.min_samples_split
    }
    
    # Train and evaluate models
    for i in range(args.repeats):
        print(f'Training iteration {i+1}/{args.repeats}')
        
        # Prepare train/test data
        X_train = np.concatenate([
            data['x'][np.isin(data['z'], names)] 
            for names in train_names
        ])
        y_train = np.concatenate([
            data['y'][np.isin(data['z'], names)]
            for names in train_names
        ])
        
        X_test = np.concatenate([
            data['x'][np.isin(data['z'], names)]
            for names in test_names
        ])
        y_test = np.concatenate([
            data['y'][np.isin(data['z'], names)]
            for names in test_names
        ])
        
        # Get test set knockout names
        knockout_test = np.concatenate([
            data['z'][np.isin(data['z'], names)]
            for names in test_names
        ])
        
        # Train and evaluate
        results_df, accuracy = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            knockout_test, params, LOCAL_DATA_FOLDER + args.savepath
        )
        
        # Save results
        results_df.to_csv(
            f'{LOCAL_DATA_FOLDER}{args.savepath}results_{i}.csv',
            index=False
        )
        print(f'Iteration {i+1} accuracy: {accuracy:.3f}')

if __name__ == '__main__':
    main()
