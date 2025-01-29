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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

# Add tools path to system path
tools_path = './tools'
sys.path.append(tools_path)
from splitting_lists import split_lists_randomly
import knockout_voting as ko

# Constants
LOCAL_DATA_FOLDER = './data/'
RANDOM_SEED = 42

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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
    """Load and prepare yeast knockout data and split into train/test sets."""
    print('Loading in Data...')
    data = np.load(LOCAL_DATA_FOLDER + 'yeast_single_knockouts.npz')
    dataframe = pd.DataFrame({'knockout': data['z'], 'essential': data['y']})
    
    # Get essential and nonessential gene lists
    nonessential_names = list(set(dataframe.loc[dataframe.essential == 1]['knockout']))
    essential_names = list(set(dataframe.loc[dataframe.essential == 0]['knockout']))
    
    # Load predefined test set
    print('Loading in predefined test set...')
    test_set = pd.read_csv(LOCAL_DATA_FOLDER + 'yeast_essentiality_test_split.csv')
    test_knockouts = test_set.loc[test_set.test == 1].knockout.unique()
    
    # Split into train/test sets
    essential_names_test = [e for e in essential_names if e in test_knockouts]
    nonessential_names_test = [e for e in nonessential_names if e in test_knockouts]
    essential_names = [e for e in essential_names if e not in test_knockouts]
    nonessential_names = [e for e in nonessential_names if e not in test_knockouts]
    
    return data, (essential_names, nonessential_names), (essential_names_test, nonessential_names_test)

def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray,
                           knockout_names: np.ndarray, model_savepath: str,
                           params: Dict, i: int = 0, fold: int = 0,
                           acc_type: str = 'knockout') -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """Train and evaluate a RandomForest model."""
    model_name = 'RandomForest'
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            min_samples_split=params['min_samples_split'],
            random_state=RANDOM_SEED))
    ])
    
    # Train and save model
    pipeline.fit(X_train, y_train)
    model_filename = f'max_depth_{params["max_depth"]}_n_estimators_{params["n_estimators"]}_min_samples_split_{params["min_samples_split"]}_{fold}_{i}_{test_split}_{model_name}.pkl'
    with open(LOCAL_DATA_FOLDER + model_savepath + model_filename, 'wb') as f:
        pickle.dump(pipeline, f)

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Prepare results DataFrame
    results_df = pd.DataFrame({
        'true_label': y_test,
        'prediction': y_pred,
        'score': y_pred_proba[:, 1],
        'model': model_name,
        'knockout_name': knockout_names
    })
    
    # Calculate knockout-level accuracy if requested
    if acc_type == 'knockout':
        accuracy = ko.calculate_knockout_accuracy(results_df, thresh=0.5, score='score', column='knockout_name')
    
    print(f'Sample-level Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Knockout-level Accuracy: {accuracy}')
    
    return pd.DataFrame(report).transpose(), accuracy, results_df

def objective(params: Dict, data: np.ndarray, folds: int) -> Dict:
    """Objective function for hyperparameter optimization."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            max_depth=params['max_depth'], 
            n_estimators=params['n_estimators'], 
            min_samples_split=params['min_samples_split'], 
            random_state=42))
    ])

    accuracy_scores = []
    all_results = pd.DataFrame()

    # Cross validation (manual to compute knockout-level)
    for fold in range(folds):
        print(f'beginning fold {fold}')
        print('Splitting training and test set...')
        essential_names_split, nonessential_names_split = split_lists_randomly(essential_names, nonessential_names, test_split)
        essential_names_val, essential_names_train = essential_names_split
        nonessential_names_val, nonessential_names_train = nonessential_names_split

        essential_data_train = data['x'][np.isin(data['z'], essential_names_train)]
        essential_data_val = data['x'][np.isin(data['z'], essential_names_val)]

        nonessential_data_train = data['x'][np.isin(data['z'], nonessential_names_train)]
        nonessential_data_val = data['x'][np.isin(data['z'], nonessential_names_val)]

        X_train = np.concatenate((essential_data_train, nonessential_data_train))
        y_train = np.concatenate((data['y'][np.isin(data['z'], essential_names_train)], data['y'][np.isin(data['z'], nonessential_names_train)]))
        X_val = np.concatenate((essential_data_val, nonessential_data_val))
        y_val = np.concatenate((data['y'][np.isin(data['z'], essential_names_val)], data['y'][np.isin(data['z'], nonessential_names_val)]))
        
        # Get knockout information for validation set
        knockout_essential = data['z'][np.isin(data['z'], essential_names_val)]
        knockout_nonessential = data['z'][np.isin(data['z'], nonessential_names_val)]
        knockout_val = np.concatenate((knockout_essential, knockout_nonessential))

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        report_df, accuracy, results_df = train_and_evaluate_model(X_train_shuffled, X_val, y_train_shuffled, y_val, knockout_val, savepath, params, i=0, fold=fold)
        results_df['fold'] = fold
        accuracy_scores.append(accuracy)
        all_results = pd.concat([all_results, results_df])
        
        # Save all results
        all_results.to_csv(LOCAL_DATA_FOLDER + savepath +  f'max_depth_{params['max_depth']}_n_estimators_{params['n_estimators']}_min_samples_split_{params['min_samples_split']}_{test_split}_all_results.csv', index=False)

    # Return the negative mean accuracy (since Hyperopt minimizes the objective)
    return {'loss': -np.mean(accuracy_scores), 'status': STATUS_OK}

def main():
    """Main execution function."""
    args = parse_arguments()
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings('ignore')
    
    # Load and prepare data
    data, (essential_names, nonessential_names), (essential_names_test, nonessential_names_test) = load_data()
    
    # Perform grid search if requested
    count = 0
    results = {}
    if args.grid_search:
        for max_depth in [None, 30]:
            for n_estimators in [100, 200, 300]:
                for min_samples_split in [2, 20]:
                    count +=1
                    print(f'({count}/12) Performing k-fold CV on parameters:', max_depth, n_estimators, min_samples_split)
                    params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_split': min_samples_split}
                    # result = objective(params, data, folds)
                    # print('Result:', result)

    

    # Set dummy parameters if not specified
    if args.max_depth is None or args.n_estimators == 'None' or args.min_samples_split == 'None':
        best = {'max_depth': None, 'n_estimators': 100, 'min_samples_split': 2}
    else:
        best = {'max_depth': args.max_depth, 'n_estimators': args.n_estimators, 'min_samples_split': args.min_samples_split}

    # Train final model with specified parameters
    for i in range(args.repeats):
        print(f'Iteration {i}')
        essential_data_train = data['x'][np.isin(data['z'], essential_names)]
        essential_data_test = data['x'][np.isin(data['z'], essential_names_test)]
        nonessential_data_train = data['x'][np.isin(data['z'], nonessential_names)]
        nonessential_data_test = data['x'][np.isin(data['z'], nonessential_names_test)]

        X_train = np.concatenate((essential_data_train, nonessential_data_train))
        y_train = np.concatenate((data['y'][np.isin(data['z'], essential_names)], data['y'][np.isin(data['z'], nonessential_names)]))
        X_test = np.concatenate((essential_data_test, nonessential_data_test))
        y_test = np.concatenate((data['y'][np.isin(data['z'], essential_names_test)], data['y'][np.isin(data['z'], nonessential_names_test)]))
        
        # Get knockout information for test set ONLY
        knockout_essential = data['z'][np.isin(data['z'], essential_names_test)]
        knockout_nonessential = data['z'][np.isin(data['z'], nonessential_names_test)]
        knockout_test = np.concatenate((knockout_essential, knockout_nonessential))

        results = []
        accuracy_scores = []
        all_results = pd.DataFrame()

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        report_df, accuracy, results_df = train_and_evaluate_model(X_train_shuffled, X_test, y_train_shuffled, y_test, knockout_test, savepath, best, i=i, fold=0)
        print('accuracy', accuracy)
        results.append(report_df)
        accuracy_scores.append(accuracy)
        all_results = pd.concat([all_results, results_df])

        # Save all results
        if folds == 0:
            all_results.to_csv(local_data_folder + savepath +  f'{i}_{test_split}_all_results.csv', index=False)
        else:
            all_results.to_csv(local_data_folder + savepath +  f'{i}_{test_split}_best_results.csv', index=False)


if __name__ == '__main__':
    main()
