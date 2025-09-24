"""
Traditional machine learning models for human activity recognition.

This module implements Random Forest and Support Vector Machine classifiers 
for activity recognition and intensity classification using extracted features 
from sensor data.
"""

import os
import random
import logging
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import joblib
import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix)


# Constants
SEED = 42
N_PARTICIPANTS = 25
SAMPLE_RATE = 60  # Hz
N_CV_FOLDS = 5
BODY_PARTS = [
    'LowerBack', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 
    'LeftShank', 'LeftFoot', 'UpperBack', 'Head', 'RightShoulder', 
    'RightUpperArm', 'RightForeArm', 'RightWrist', 'LeftShoulder', 
    'LeftUpperArm', 'LeftForeArm', 'LeftWrist'
]

# Feature selection configuration
N_FEATURES = 18
DROPPED_FEATURE_COUNTS = {
    0: [],  # Keep all features
    9: ['range', 'skewness', 'energy', 'entropy', 'iqr', 'mad', 'rms', 'sma'],
    11: ['range', 'skewness', 'energy', 'entropy', 'iqr', 'mad', 'rms', 'sma',
         'zero_crossing_rate', 'mean_crossing_rate'],
    14: ['range', 'skewness', 'energy', 'entropy', 'iqr', 'mad', 'rms', 'sma',
         'zero_crossing_rate', 'mean_crossing_rate', 'dominant_frequency', 'max', 'min']
}


def setup_logger(log_path, verbosity=1, name=None):
    """
    Configure logging for training process.
    
    Args:
        log_path: Path for log file
        verbosity: Logging level (0=DEBUG, 1=INFO, 2=WARNING)
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # File handler
    file_handler = logging.FileHandler(log_path, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def compute_intensity_labels(activity_labels):
    """
    Convert activity labels to intensity categories.

    Note:
        - Current activity_labels and intensity_labels are 1-indexed (1, 2, ..., 12)
        - We later convert them to 0-indexed for model training

    Args:
        activity_labels: Array of activity labels (1-indexed)

    Returns:
        intensity_labels: Array of intensity categories (1-indexed)
    """
    intensity_labels = activity_labels.copy()
    intensity_labels[(activity_labels == 1) | (activity_labels == 2)] = 1  # sedentary
    intensity_labels[(activity_labels == 3) | (activity_labels == 4)] = 2  # light
    intensity_labels[(activity_labels == 5) | (activity_labels == 6) | 
                    (activity_labels == 8) | (activity_labels == 9) | 
                    (activity_labels == 12)] = 3  # moderate
    intensity_labels[(activity_labels == 7) | (activity_labels == 10) | 
                    (activity_labels == 11)] = 4  # vigorous  
    return intensity_labels


def load_features(feature_path, window_size, n_dropped_features,
                                logger, dropped_body_parts):
    """
    Load feature data for all participants.
    
    Args:
        feature_path: Path to features directory
        window_size: Window size in seconds
        n_dropped_features: Number of features to drop (0, 9, 11, or 14)
        dropped_body_parts: List of body part names to exclude
        logger: Logger instance
        
    Returns:
        List of feature arrays for each participant
    """
    window_dir = os.path.join(feature_path, f'{window_size}s')
    all_features = []
    
    logger.info(f'Loading data from: {window_dir}')
    
    for participant_id in range(N_PARTICIPANTS):
        data_file = f'features_P{participant_id+1:02d}.csv'
        file_path = os.path.join(window_dir, data_file)
        
        if not os.path.exists(file_path):
            logger.warning(f'File not found: {file_path}')
            continue
            
        participant_features = pd.read_csv(file_path)
        
        logger.info(f'Original feature shape for {data_file}: {participant_features.shape}')
        
        # Drop specific features based on configuration
        if n_dropped_features > 0 and n_dropped_features in DROPPED_FEATURE_COUNTS:
            features_to_drop = DROPPED_FEATURE_COUNTS[n_dropped_features]
            feature_columns_to_drop = [
                col for col in participant_features.columns 
                if any(f'_{feature}' in col for feature in features_to_drop)
            ]
            participant_features = participant_features.drop(columns=feature_columns_to_drop)
            logger.info(f'After dropping {n_dropped_features} feature(s), feature shape: {participant_features.shape}')
        
        # Drop body part features if specified
        if len(dropped_body_parts) > 0:
            body_part_columns_to_drop = [
                col for col in participant_features.columns 
                if any(body_part in col for body_part in dropped_body_parts)
            ]
            participant_features = participant_features.drop(columns=body_part_columns_to_drop)
            logger.info(f'After dropping {len(dropped_body_parts)} body part(s), feature shape: {participant_features.shape}')
        
        # Clean data: handle NaN and infinite values
        if participant_features.isnull().values.any():
            logger.warning(f'NaN values found in {data_file}')
            participant_features = participant_features.dropna()
            
        if np.isinf(participant_features.values).any():
            logger.warning(f'Infinite values found in {data_file}')
            participant_features = participant_features.replace([np.inf, -np.inf], np.nan)
            participant_features = participant_features.dropna()
        
        all_features.append(participant_features.values)
    
    return all_features


def load_raw_data(data_path, window_size, overlap = 0.5):
    """
    Load raw sensor data from numpy files.
    
    Args:
        data_path: Path to raw data directory
        window_size: Window size in seconds
        overlap: Overlap ratio between consecutive windows
        
    Returns:
        Tuple of (X, Y_act, Y_int) lists
    """
    window_subdir = f'{window_size}s'
    
    X, Y_act, Y_int = [], [], []
    for participant_id in range(N_PARTICIPANTS):
        # Load preprocessed numpy arrays
        x_path = os.path.join(data_path, window_subdir, f'P{participant_id+1:02d}_X.npy')
        y_act_path = os.path.join(data_path, window_subdir, f'P{participant_id+1:02d}_Y_act.npy')
        y_int_path = os.path.join(data_path, window_subdir, f'P{participant_id+1:02d}_Y_int.npy')

        missing_files = [path for path in [x_path, y_act_path, y_int_path] if not os.path.exists(path)]
        to_process_raw_data = (len(missing_files) > 0)
        logging.info(f'{x_path},{y_act_path},{y_int_path},{to_process_raw_data}')
        if to_process_raw_data:
            participant_X, participant_Y_act, participant_Y_int = [], [], []
            
            # Process raw CSV data
            file_path = os.path.join(data_path, f'P{participant_id+1:02d}.csv')
            data = pd.read_csv(file_path)
            logger.info(f'Processing {file_path}')
            
            # Remove quaternion columns (keep only Acc and Gyr: 6*17=102 channels)
            quaternion_columns = [col for col in data.columns if 'Quat' in col]
            data = data.drop(columns=quaternion_columns)

            # Apply sliding windows
            n_timesteps = len(data)
            window_samples = int(window_size * SAMPLE_RATE) # Number of data samples in each sliding window
            step_samples = int(window_samples * (1 - overlap)) # Step in samples

            for i in range(0, n_timesteps - window_samples + 1, step_samples):
                window_data = data.iloc[i:i + window_samples]
                activity_labels = np.unique(window_data['Activity'].values)
                
                # Skip windows with multiple activities or invalid labels (0)
                if len(activity_labels) > 1 or activity_labels[0] == 0:
                    continue
                
                window_data = window_data.drop(columns=['Activity'])

                participant_X.append(window_data)
                participant_Y_act.append(activity_labels[0])
                participant_Y_int.append(compute_intensity_labels(activity_labels)[0])

        else:
            logger.info(f'Loading data from: {x_path}, {y_act_path}, and {y_int_path}')
            # Reshape sensor data to flatten time series
            participant_X = np.load(x_path).reshape(-1, int(17 * 6 * window_size * SAMPLE_RATE))
            participant_Y_act = np.load(y_act_path)
            participant_Y_int = np.load(y_int_path)

    
        if to_process_raw_data:
            # Convert to arrays and adjust labels (1-indexed to 0-indexed)
            participant_X = np.array(participant_X).reshape(-1, int(17 * 6 * window_size * SAMPLE_RATE))
            participant_Y_act = (np.array(participant_Y_act) - 1).astype(np.int64)
            participant_Y_int = (np.array(participant_Y_int) - 1).astype(np.int64)
            
            # Save preprocessed data
            npy_dir = os.path.join(os.path.dirname(file_path), window_subdir)
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)
            np.save(x_path, participant_X)
            np.save(y_act_path, participant_Y_act)
            np.save(y_int_path, participant_Y_int)
        
        X.append(participant_X)
        Y_act.append(participant_Y_act)
        Y_int.append(participant_Y_int)

    return X, Y_act, Y_int


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  model_name: str, fold_idx: int, logger: logging.Logger) -> dict:
    """
    Evaluate model performance and return metrics.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for logging
        fold_idx: Cross-validation fold index
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # Log results
    logger.info(f'CV {fold_idx} - {model_name} Accuracy: {accuracy*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} Precision (macro): {macro_precision*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} Recall (macro): {macro_recall*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} F1-Score (macro): {macro_f1*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} Precision (weighted): {weighted_precision*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} Recall (weighted): {weighted_recall*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} F1-Score (weighted): {weighted_f1*100:.6f}%')
    logger.info(f'CV {fold_idx} - {model_name} Confusion Matrix:\n{confusion_mat}')
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': confusion_mat
    }


def save_model_results(results_dir, window_subdir, task,
                      model_name, fold_idx, metrics, model=None):
    """
    Save model results and trained model to disk.
    
    Args:
        results_dir: Base results directory
        window_subdir: Window size subdirectory
        task: Task type ('activity' or 'intensity')
        model_name: Model name ('rf' or 'svm')
        fold_idx: Cross-validation fold index
        metrics: Dictionary of evaluation metrics
        model: Trained model to save (optional)
    """
    # Create output directory
    output_dir = os.path.join(results_dir, window_subdir, task)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual fold results
    for metric_name, metric_value in metrics.items():
        save_path = os.path.join(output_dir, f'{model_name}_confusion_matrix_fold{fold_idx}.npy')
        np.save(save_path, metric_value)
    
    # Save trained model
    if model is not None:
        model_path = os.path.join(output_dir, f'{model_name}_fold_{fold_idx}.pkl')
        joblib.dump(model, model_path)


def train_traditional_models(data_path, window_size, logger, results_path,
                             tasks=['activity', 'intensity'], 
                             use_raw_data = False, n_dropped_features = 0,
                             dropped_body_parts = []):
    """
    Train and evaluate traditional machine learning models.
    
    Args:
        data_path: Path to dataset directory
        window_size: Window size in seconds
        logger: Logger instance
        results_path: Output directory for results
        use_raw_data: Whether to use raw sensor data or extracted features
        n_dropped_features: Number of feature types to drop
        dropped_body_parts: List of body parts to exclude
    """
    window_subdir = f'{window_size}s'
    logger.info(f'Training models for window size: {window_size}s')
    logger.info(f'Data directory: {data_path}')
    
    # Configure results directory based on experimental setup
    if not use_raw_data:
        if not dropped_body_parts:
            results_path = os.path.join(results_path, f'results_{N_FEATURES-n_dropped_features}')
        elif len(dropped_body_parts) == 16:
            results_dir = os.path.join(results_dir,'results_body_part{}'.format(list(set(BODY_PARTS)-set(dropped_body_parts))[0]))
        else:
            n_remaining_parts = len(BODY_PARTS) - len(dropped_body_parts)
            results_dir = os.path.join(results_dir,'results_body_part{}'.format(n_remaining_parts))
    else:
        results_path = os.path.join(results_path, 'results_raw')
    
    logger.info(f'Results will be saved to: {results_path}')
    
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # Initialize models
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    svm = SVC(random_state=SEED)
    if use_raw_data:
        models = [svm]
        model_names = ['svm']
    else:
        models = [rf, svm]
        model_names = ['rf', 'svm']
    
    # Load data based on type
    if use_raw_data:
        X, Y_act_data, Y_int_data = load_raw_data(data_path, window_size)
        logger.info(f'Loaded raw data for {len(X)} participants')
        assert len(X) == N_PARTICIPANTS, 'Incorrect number of participants'
    else:
        all_features = load_features(
            data_path, window_size, n_dropped_features,
            logger, dropped_body_parts
        )
        logger.info(f'Loaded feature data for {len(all_features)} participants')
        assert len(all_features) == N_PARTICIPANTS, 'Incorrect number of participants'

    # Setup cross-validation
    kfold = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    cv_splits = list(kfold.split(np.arange(N_PARTICIPANTS)))
    
    # Train and evaluate for each task
    for task in tasks:
        logger.info(f'Processing task: {task}')

        for fold_idx, (train_indices, test_indices) in tqdm.tqdm(enumerate(cv_splits, 1)):
            logger.info(f"=== Task: {task}, Window: {window_subdir}, Fold: {fold_idx} ===")
            logger.info(f"Train subjects: {train_indices}, Test subjects: {test_indices}")
            
            # Prepare training and test data
            if use_raw_data:
                # Use raw sensor data
                train_x = [X[i] for i in train_indices]
                test_x = [X[i] for i in test_indices]
                train_x = np.concatenate(train_x, axis=0)
                test_x = np.concatenate(test_x, axis=0)
                
                if task == 'intensity':
                    train_y = [Y_int_data[i] for i in train_indices]
                    test_y = [Y_int_data[i] for i in test_indices]
                else:  # activity
                    train_y = [Y_act_data[i] for i in train_indices]
                    test_y = [Y_act_data[i] for i in test_indices]
                
                train_y = np.concatenate(train_y).flatten()
                test_y = np.concatenate(test_y).flatten()
                
            else:
                # Use extracted features
                train_data = [all_features[i] for i in train_indices]
                test_data = [all_features[i] for i in test_indices]
                train_data = np.concatenate(train_data, axis=0)
                test_data = np.concatenate(test_data, axis=0)
                
                # Extract features and labels
                train_x = train_data[:, 1:]  # All columns except first (activity label)
                train_y_act = train_data[:, 0]  # First column (activity labels)
                train_y_int = compute_intensity_labels(train_y_act)
                
                test_x = test_data[:, 1:]
                test_y_act = test_data[:, 0]
                test_y_int = compute_intensity_labels(test_y_act)
                
                # Select labels based on task
                train_y = train_y_int if task == 'intensity' else train_y_act
                test_y = test_y_int if task == 'intensity' else test_y_act
            
            logger.info(f'Train data shape: {train_x.shape}, Test data shape: {test_x.shape}')
            logger.info(f'Train labels shape: {train_y.shape}, Test labels shape: {test_y.shape}')
            
            # Standardize features
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
            
            # Train Models
            for model, model_name in zip(models, model_names):
                model.fit(train_x, train_y)
                
                # Evaluate
                metrics = evaluate_model(
                    model, test_x, test_y, model_name, fold_idx, logger
                )
                
                # Save results
                save_model_results(
                    results_path, window_subdir, task, model_name, 
                    fold_idx, metrics, model
                )

        logger.info(f"Completed task: {task} for window size: {window_size}s")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train traditional machine learning models for HAR')
    parser.add_argument('--feature_data_path', type=str, default='data/features',
                   help='Path to extracted features directory')
    parser.add_argument('--raw_data_path', type=str, default='data/dataset',
                    help='Path to raw dataset directory')
    parser.add_argument('--results_path', type=str, default='baseline_v1/traditional/',
                    help='Path to save results')
    parser.add_argument('--log_path', type=str, default='baseline_v1/traditional/logs/',
                    help='Path to save logs')
    
    args = parser.parse_args()

    # Configure logging
    log_path = args.log_path
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    results_path = args.results_path
    
    logger = setup_logger(os.path.join(log_path, f'{current_time}_logger.log'))

    # Dataset configuration
    feature_data_path = args.feature_data_path
    raw_data_path = args.raw_data_path
    
    # Experimental configurations
    window_sizes = [10, 7.5, 5, 2.5, 2, 1.5, 1, 0.5]  # seconds
    feature_drop_configs = [0, 9, 11, 14]  # Number of feature to drop

    # Run experiments
    for window_size in window_sizes:  # Effect of window sizes
        logger.info(f"=" * 50)
        logger.info(f"Processing window size: {window_size}s")
                
        # Train with extracted features
        for n_dropped_features in feature_drop_configs: # Effect of feature engineering
            train_traditional_models(
                data_path=feature_data_path,
                window_size=window_size,
                logger=logger,
                results_path=results_path,
                use_raw_data = False,
                n_dropped_features=n_dropped_features
            )

    window_sizes_for_raw_data = [10, 7.5, 5, 2.5]  # seconds
    for window_size in window_sizes_for_raw_data:  # Effect of window sizes
        logger.info(f"=" * 50)
        logger.info(f"Processing window size: {window_size}s")

        # Train with raw data
        train_traditional_models(
            data_path=raw_data_path,
            window_size=window_size,
            logger=logger,
            results_path=results_path,
            use_raw_data=True
        )
    
    # Define sensor configurations
    full_body = [
        'LowerBack', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 
        'LeftShank', 'LeftFoot', 'UpperBack', 'Head', 'RightShoulder', 
        'RightUpperArm', 'RightForeArm', 'RightWrist', 'LeftShoulder', 
        'LeftUpperArm', 'LeftForeArm', 'LeftWrist'
    ]
    upper_body = [
        'UpperBack', 'Head', 'RightShoulder', 
        'RightUpperArm', 'RightForeArm', 'RightWrist', 'LeftShoulder', 
        'LeftUpperArm', 'LeftForeArm', 'LeftWrist'
    ]
    lower_body = [
        'LowerBack', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 
        'LeftShank', 'LeftFoot'
    ]
    l5_thigh_shank = ['LowerBack', 'LeftThigh', 'LeftShank']
    left_thigh = ['LeftThigh']
    left_foot = ['LeftFoot']
    left_wrist = ['LeftWrist']
    
    for sensor_configuration in [full_body, upper_body, lower_body, l5_thigh_shank,
                                 left_thigh, left_foot, left_wrist]:
        dropped_body_parts = list(set(BODY_PARTS) - set(sensor_configuration))
        
        logger.info(f'Dropping {len(dropped_body_parts)} body parts: {dropped_body_parts}')

        window_sizes = [2]  # seconds
        for window_size in window_sizes:
            # Train with all 18 extracted features
            train_traditional_models(
                data_path=feature_data_path,
                window_size=window_size,
                logger=logger,
                results_path=results_path,
                use_raw_data=False
            )
    

