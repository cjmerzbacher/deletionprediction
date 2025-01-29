"""
Train and evaluate RandomForest models for E. coli essentiality classification.
Uses train/test splits and saves models and results.
"""

import argparse
import pickle
import random
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
LOCAL_DATA_FOLDER = './data/'

# Model-specific biomass column indices to remove from features
BIOMASS_INDICES = {
    'iml1515': [2669, 2670],
    'iaf1260': [926],
    'ijr904': [269],
    'ijo1366': [19, 14]
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model configuration."""
    parser = argparse.ArgumentParser(
        prog='Trainer',
        description='Trains RandomForest models for E. coli essentiality prediction'
    )
    parser.add_argument('--repeats', type=int, help='Number of training repeats')
    parser.add_argument('--test_split', type=float, help='Test set fraction (e.g. 0.2 for 20% test)')
    parser.add_argument('--savepath', help='Path to save model and results')
    parser.add_argument('--model', help='E. coli model name')
    return parser.parse_args()

def load_data(model: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load essential and nonessential knockout data for given model."""
    # Load data files
    essential_data = np.load(f'{LOCAL_DATA_FOLDER}{model}_1x_essential.npz')
    nonessential_data = np.load(f'{LOCAL_DATA_FOLDER}{model}_1x_nonessential.npz')
    
    # Get unique knockout names
    essential_names = list(set(essential_data['z']))
    nonessential_names = list(set(nonessential_data['z']))
    
    return essential_data, nonessential_data, essential_names, nonessential_names

def split_data_randomly(list1: List, list2: List, split_percentage: float) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """Split two lists randomly based on split percentage."""
    if not (0 < split_percentage < 1):
        raise ValueError("Split percentage must be between 0 and 1")

    # Create copies and shuffle
    random.seed()
    list1_copy = list1[:]
    list2_copy = list2[:]
    random.shuffle(list1_copy)
    random.shuffle(list2_copy)

    # Calculate split indices
    split_index1 = int(len(list1_copy) * split_percentage)
    split_index2 = int(len(list2_copy) * split_percentage)

    # Split lists
    list1_splits = (list1_copy[:split_index1], list1_copy[split_index1:])
    list2_splits = (list2_copy[:split_index2], list2_copy[split_index2:])

    return list1_splits, list2_splits

def prepare_train_test_data(essential_data: Dict, nonessential_data: Dict,
                          train_names: Tuple[List, List], test_names: Tuple[List, List],
                          model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and test datasets."""
    # Extract train/test names
    essential_train, nonessential_train = train_names
    essential_test, nonessential_test = test_names
    
    # Create masks
    train_mask = np.isin(essential_data['z'], essential_train)
    test_mask = np.isin(essential_data['z'], essential_test)
    
    # Combine data
    X_train = np.concatenate((
        essential_data['x'][train_mask],
        nonessential_data['x'][np.isin(nonessential_data['z'], nonessential_train)]
    ))
    X_test = np.concatenate((
        essential_data['x'][test_mask],
        nonessential_data['x'][np.isin(nonessential_data['z'], nonessential_test)]
    ))
    
    y_train = np.concatenate((
        essential_data['y'][train_mask],
        nonessential_data['y'][np.isin(nonessential_data['z'], nonessential_train)]
    ))
    y_test = np.concatenate((
        essential_data['y'][test_mask],
        nonessential_data['y'][np.isin(nonessential_data['z'], nonessential_test)]
    ))
    
    # Remove biomass columns if needed
    if model_name.lower() in BIOMASS_INDICES:
        indices = BIOMASS_INDICES[model_name.lower()]
        X_train = np.delete(X_train, indices, 1)
        X_test = np.delete(X_test, indices, 1)
        
    # Get test set knockout names
    knockout_test = np.concatenate((
        essential_data['z'][test_mask],
        nonessential_data['z'][np.isin(nonessential_data['z'], nonessential_test)]
    ))
    
    return X_train, X_test, y_train, y_test, knockout_test

def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray, 
                           knockout_names: np.ndarray, model_savepath: str,
                           iteration: int, test_split: float) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """Train RandomForest model and evaluate performance."""
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    pipeline.fit(X_train, y_train)
    
    # Save trained model
    model_path = f'{LOCAL_DATA_FOLDER}{model_savepath}{iteration}_{test_split}_RandomForest.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'true_label': y_test,
        'prediction': y_pred,
        'score': y_pred_proba[:, 1],
        'model': 'RandomForest',
        'knockout_name': knockout_names
    })
    
    return report_df, accuracy, results_df

def main():
    """Main training loop."""
    args = parse_arguments()
    warnings.filterwarnings('ignore')
    
    # Load data
    print('Loading data...')
    essential_data, nonessential_data, essential_names, nonessential_names = load_data(args.model)
    
    # Run training iterations
    for i in range(args.repeats):
        print(f'Iteration {i}')
        
        # Split data
        print('Splitting data...')
        essential_splits, nonessential_splits = split_data_randomly(
            essential_names, nonessential_names, float(args.test_split)
        )
        
        # Prepare datasets
        X_train, X_test, y_train, y_test, knockout_test = prepare_train_test_data(
            essential_data, nonessential_data,
            (essential_splits[1], nonessential_splits[1]),
            (essential_splits[0], nonessential_splits[0]),
            args.model
        )
        
        # Shuffle training data
        print('Training model...')
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Train and evaluate
        _, accuracy, results_df = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            knockout_test, args.savepath, i, args.test_split
        )
        print(f'Accuracy: {accuracy}')

        # Save results
        results_path = f'{LOCAL_DATA_FOLDER}{args.savepath}{i}_{args.test_split}_all_results.csv'
        results_df.to_csv(results_path, index=False)

if __name__ == '__main__':
    main()
