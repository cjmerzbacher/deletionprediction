# Deletion Prediction

This repository contains code for training and evaluating machine learning models to predict the effects of gene deletions in different organisms.

## Repository Structure

### Environment and Dependencies
The `environment.yml` file contains all dependencies needed to run the code. It can be installed in a fresh Conda environment using the following command: `conda env create -f environment.yml`.

### Figure Creation Notebooks

The data and code to create all figures in the paper "Accurate prediction of gene deletion phenotypes with Flux Cone Learning" is organized by figure panel. Some figures require multiple data files; Figures 2A and B are created from the same dataset. 

### Training Scripts

- `training/ecoli_training.py`: Trains RandomForest models for E. coli essentiality classification using train/test splits. Note that this script trains models on all reactions in a given GEM, not only the shared reactions as in Figure 1F.
- `training/yeast_training_esseniality.py`: Trains RandomForest models for yeast essentiality classification using k-fold cross validation and hyperparameter optimization
- `training/yeast_training_production.py`: Trains multiple ML models (HistGradientBoosting, LinearSVC, LogisticRegression, RandomForest) for yeast production classification with balanced and resampled variants
- `training/cho_training.py`: Trains HistGradientBoosting models for CHO cell essentiality classification using k-fold cross validation and hyperparameter optimization

### Data

The training data should be placed in the `data/` directory and includes:

- CHO cell data: `cho_essential_full.npz`, `cho_nonessential_full.npz`
- E. coli data: Model-specific files like `iml1515_1x_essential.npz`
- Yeast data: `yeast_single_knockouts.npz`

The `data/` directory also contains several CSV files used for data splits and analysis:

- `groundtruthcomplete.csv`: Includes the ground truth labels for E. coli gene essentiality.
- `yeast_essentiality_test_split.csv`: Defines the test set for yeast data by marking knockouts as test/non-test
- `yeast_production_test_split.csv`: Lists knockout names designated for the test set in yeast production prediction
- `yeast_production_validation_split.csv`: Contains k-fold validation splits for yeast production data, with knockout names and fold assignments
- `cho_essentiality_validation_split.csv`: Contains k-fold validation splits for CHO cell data, with columns for knockout names and fold assignments
- `cho_essentiality_test_split.csv`: Defines the test set for CHO cell data by marking knockouts as test/non-test

### Model Parameters

The training scripts accept various command line arguments to configure:

- Number of training repeats/folds
- Test set split percentage 
- Model hyperparameters (learning rate, tree depth, etc.)
- Paths for saving models and results

## Usage

Example usage for training an E. coli model:

### Basic usage with default parameters
python ecoli_training.py --model 'iml1515' --savepath 'results/' --repeats 5 --test_split 0.2

### The script will:
1. Load E. coli knockout data for iml1515 model
2. Split data into train/test sets (20% test)
3. Train a RandomForest model 5 times with different random splits
4. Save models and results to models/ directory

Example usage for training a yeast essentiality model:

### Basic usage with default parameters:
python yeast_training_essentiality.py --savepath 'results/'

### Full parameter specification:
python yeast_training_essentiality.py \
    --savepath 'results/' \
    --repeats 5 \
    --folds 5 \
    --grid_search True \
    --max_depth 10 \
    --n_estimators 200 \
    --min_samples_split 5

The script will:
1. Load yeast knockout data
2. Split data into train/test sets based on predefined splits
3. Train RandomForest models with specified parameters
4. Evaluate using k-fold cross validation
5. Save models and results to specified path

### For hyperparameter tuning:
python yeast_training_essentiality.py \
    --savepath 'tuning/' \
    --grid_search True \
    --repeats 3

Example usage for training a yeast production model suite:

### Basic usage with default parameters:
python yeast_training_production.py --savepath 'results/'

### Full parameter specification:
python yeast_training_production.py \
    --savepath 'results/' \
    --repeats 5 \
    --folds 5 \
    --test_split 0.2

### The script will:
1. Load yeast knockout data and production values
2. Preprocess data by removing NaNs and scaling production values
3. Bin production into 3 classes (low/medium/high)
4. Train multiple ML models with k-fold cross validation
5. Save models and results to specified path

Example usage for training a CHO model:

### Basic usage with default parameters
python cho_training.py --savepath 'results/' --max_iter 100 --learning_rate 0.1

### Full parameter specification
python cho_training.py \
    --savepath 'results/' \
    --repeats 5 \
    --test_split 0.2 \
    --num_samples 100 \
    --downsample 1 \
    --max_depth 10 \
    --max_iter 100 \
    --learning_rate 0.1

### For hyperparameter tuning
python cho_training.py \
    --savepath tuning/ \
    --max_depth 5 \
    --max_iter 50 \
    --learning_rate 0.05

The script will:
1. Load CHO cell knockout data
2. Split data into train/test sets
3. Train a HistGradientBoosting model with specified parameters
4. Evaluate using k-fold cross validation
5. Save model and results to specified path


