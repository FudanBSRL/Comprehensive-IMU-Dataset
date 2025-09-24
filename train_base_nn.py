"""
Deep learning models for human activity recognition using sensor data.

This module implements ResNet1D and LSTM architectures for classification 
of human activities and activity intensities.
"""

import os
import glob
import json
import random
import logging
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Constants
SEED = 42
N_PARTICIPANTS = 25
N_SENSORS = 17
N_CHANNELS_PER_SENSOR = 6
SAMPLE_RATE = 60  # Hz
N_CV_FOLDS = 5
BODY_PARTS = [
    'LowerBack', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 
    'LeftShank', 'LeftFoot', 'UpperBack', 'Head', 'RightShoulder', 
    'RightUpperArm', 'RightForeArm', 'RightWrist', 'LeftShoulder', 
    'LeftUpperArm', 'LeftForeArm', 'LeftWrist'
]


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
    

class HARDataset(Dataset):
    """
    Dataset class for Human Activity Recognition data with sliding window preprocessing.
    
    Loads raw sensor data, applies sliding windows, and converts activity labels
    to intensity categories.
    """
    
    def __init__(self, file_list, window_size, overlap, dropped_channels=[], logger=None):
        """
        Initialize HAR dataset.
        
        Args:
            file_list: List of CSV file paths for participants
            window_size: Window size in seconds
            overlap: Overlap ratio between consecutive windows
            dropped_channels: List of channels to exclude
            logger: Logger instance
        """
        self.window_size = window_size
        self.window_samples = int(window_size * SAMPLE_RATE) # Number of data samples in each sliding window
        self.overlap = overlap
        self.dropped_channels = dropped_channels
        self.logger = logger
        
        self.X, self.Y_act, self.Y_int = [], [], [] # data, labels_activity, labels_intensity 
        self._load_and_preprocess(file_list)

    def _load_and_preprocess(self, file_list):
        """Load data files and apply sliding window preprocessing."""
        
        step_samples = int(self.window_samples * (1 - self.overlap)) # Step in samples
        
        for file_path in file_list:
            # Check for preprocessed numpy files
            npy_dir = os.path.join(os.path.dirname(file_path), f'{self.window_size}s')
            base_name = os.path.basename(file_path).replace('.csv', '')
            
            x_path = os.path.join(npy_dir, f'{base_name}_X.npy')
            y_act_path = os.path.join(npy_dir, f'{base_name}_Y_act.npy')
            y_int_path = os.path.join(npy_dir, f'{base_name}_Y_int.npy')
            
            missing_files = [path for path in [x_path, y_act_path, y_int_path] if not os.path.exists(path)]
            to_process_raw_data = (len(missing_files) > 0)
            # logging.info(f'{x_path},{y_act_path},{y_int_path},{to_process_raw_data}')
            if to_process_raw_data:
                if logger:
                    logger.info(f'Processing {file_path}')
                # Process raw CSV data
                participant_X, participant_Y_act, participant_Y_int = [], [], []
                data = pd.read_csv(file_path)
                
                # Remove quaternion columns (keep only Acc and Gyr: 6*17=102 channels)
                quaternion_columns = [col for col in data.columns if 'Quat' in col]
                data = data.drop(columns=quaternion_columns)

                # Apply sliding windows
                n_timesteps = len(data)
                for i in range(0, n_timesteps - self.window_samples + 1, step_samples):
                    window_data = data.iloc[i:i + self.window_samples]
                    activity_labels = np.unique(window_data['Activity'].values)
                    
                    # Skip windows with multiple activities or invalid labels (0)
                    if len(activity_labels) > 1 or activity_labels[0] == 0:
                        continue
                    
                    window_data = window_data.drop(columns=['Activity'])

                    participant_X.append(window_data)
                    participant_Y_act.append(activity_labels[0])
                    participant_Y_int.append(compute_intensity_labels(activity_labels)[0])
                
                # Convert to arrays and adjust labels (1-indexed to 0-indexed)
                participant_X = np.array(participant_X)# (n_windows, window_len, 6*17)
                participant_Y_act = (np.array(participant_Y_act) - 1).astype(np.int64)
                participant_Y_int = (np.array(participant_Y_int) - 1).astype(np.int64)
                
                # Save preprocessed data
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)
                np.save(x_path,participant_X)
                np.save(y_act_path, participant_Y_act)
                np.save(y_int_path, participant_Y_int)

            else:
                # Load preprocessed data
                logger.info(f'Loading data from: {x_path}, {y_act_path}, and {y_int_path}')
                # Reshape sensor data to flatten time series
                participant_X = np.load(x_path)# (n_windows, window_len, 6*17)
                participant_Y_act = np.load(y_act_path)
                participant_Y_int = np.load(y_int_path)
            
            # Apply body part filtering if specified
            if len(self.dropped_channels) > 0:
                participant_X = np.delete(participant_X, self.dropped_channels, axis=2)
                
            self.X.append(participant_X)
            self.Y_act.append(participant_Y_act)
            self.Y_int.append(participant_Y_int)

        # del X, Y_act, Y_int
        # Concatenate data from all participants
        self.X = np.concatenate(self.X, axis=0)
        self.Y_act = np.concatenate(self.Y_act, axis=0)
        self.Y_int = np.concatenate(self.Y_int, axis=0)
        
        # Compute normalization statistics
        self.sample_mean = np.mean(self.X, axis=0)
        self.sample_std = np.std(self.X, axis=0)

    def normalize_data(self, sample_mean=None, sample_std=None):
        """Normalize dataset using provided or computed statistics."""
        if sample_mean is None or sample_std is None:
            sample_mean = self.sample_mean
            sample_std = self.sample_std
        self.X = (self.X - sample_mean) / sample_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Get single data sample."""
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_act = torch.tensor(self.Y_act[idx], dtype=torch.int64)
        y_int = torch.tensor(self.Y_int[idx], dtype=torch.int64)
        return x, y_act, y_int


# ResNet Building Blocks

class BasicBlock1D(nn.Module):
    """Basic residual block for ResNet18/34 architectures."""
    
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channels)
        
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class Bottleneck1D(nn.Module):
    """Bottleneck residual block for ResNet50/101 architectures."""
    
    expansion = 4

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channels)
        
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)
        
        self.conv3 = nn.Conv1d(output_channels, output_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(output_channels * self.expansion)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


# ResNet Architectures

class ResNet1D(nn.Module):
    """1D ResNet architecture for time-series classification."""

    def __init__(self, block, layers, input_channels=102, num_classes=12):
        """
        Initialize ResNet1D model.
        
        Args:
            block: BasicBlock1D or Bottleneck1D
            layers: List of block counts for each layer
            input_channels: Number of input sensor channels
            num_classes: Number of output classes
        """
        super().__init__()
        self.input_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        """Create a residual layer with specified number of blocks."""
        downsample = None
        if stride != 1 or self.input_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.input_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels * block.expansion)
            )

        layers = [block(self.input_channels, channels, 
                        stride, downsample)]
        self.input_channels = channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.input_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         # (B, channels, 1)
        x = torch.flatten(x, 1)     # (B, channels)
        return self.fc(x)


class LSTM(nn.Module):
    """LSTM model for sequential data classification."""

    def __init__(self, input_channels, num_classes, hidden_size=128, 
                 num_layers=2, bidirectional=True):
        """
        Initialize LSTM model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_channels, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(0.5)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # Shape: (batch, time, hidden*directions)
        out = out[:, -1, :]  # Use last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Training and Evaluation Functions

def train_epoch(model, dataloader, loss_fn, optimizer, device, task):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        loss_fn: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm
        device: Computing device (cpu/cuda)
        task: 'activity' or 'intensity'
        
    Returns:
        tuple: (loss, accuracy)
    """
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for x, y_act, y_int in dataloader:
        # Select target based on task type
        y = y_act if task == 'activity' else y_int
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    running_loss = running_loss / total
    accuracy = correct / total
    return running_loss, accuracy


def evaluate_epoch(model, dataloader, loss_fn, device, task):
    """
    Evaluate model performance on validation/test set.
    
    Args:
        model: Neural network model
        dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Computing device
        task: 'activity' or 'intensity'
        
    Returns:
        tuple: (loss, accuracy, precision, recall, f1_score)
    """
    model.eval()
    y_true, y_pred = [], []
    test_loss = 0
    
    with torch.no_grad():
        for x, y_act, y_int in dataloader:
            y = y_act if task == 'activity' else y_int
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            preds = logits.argmax(dim=1)
            loss = loss_fn(logits, y)
            
            test_loss += loss.item()
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    
    # Compute metrics
    test_loss = test_loss / len(dataloader)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    return test_loss, accuracy, precision, recall, f1_score


def run_cross_validation(data_path, window_size, overlap, model_cls, device, 
                        kf, dropped_channels=[], tasks=['activity', 'intensity'],
                        log_path='/tmp/har_logs', batch_size=32, 
                        n_channels=102, epochs=50, mnt_metric='valid_accuracy', 
                        early_stop_patience=10, lr=1e-3, 
                        logging_time=None, logger=None, config_args=None):
    """
    Run subject-wise k-fold cross-validation training and evaluation.
    
    Args:
        data_path: Path to dataset directory
        window_size: Window size in seconds
        overlap: Window overlap ratio
        model_cls: Model class to instantiate
        device: Computing device
        kf: KFold cross-validation splits
        dropped_channels: List of channels to exclude
        tasks: List of tasks ('activity', 'intensity')
        log_path: Directory for logging results
        batch_size: Training batch size
        n_channels: Number of input channels
        epochs: Maximum training epochs
        mnt_metric: Metric for early stopping
        early_stop_patience: Early stopping patience
        lr: Learning rate
        logging_time: Timestamp for logging
        logger: Logger instance
        config_args: Additional configuration arguments
    """
    assert n_channels == ( N_CHANNELS_PER_SENSOR * N_SENSORS - len(dropped_channels))
    # Load all participant files
    data_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    assert len(data_files) == 25, f"Expected 25 participants, found {len(data_files)}"
    
    # Define result columns for tracking metrics
    result_columns = ['epoch', 'train_loss', 'train_accuracy', 
                     'valid_loss', 'valid_accuracy', 'valid_precision', 
                     'valid_recall', 'valid_fscore']
    
    tasks = tasks
    
    for task in tasks:
        best_metrics_per_fold = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(kf, 1):
            logger.info(f"=== Task: {task}, Window: {window_size}s, "
                       f"Model: {model_cls.__name__}, Fold: {fold_idx} ===")
            logger.info(f"Train subjects: {train_indices}, Test subjects: {test_indices}")
            
            # Setup result tracking
            results_df = pd.DataFrame(columns=result_columns)
            checkpoint_dir = os.path.join(
                log_path, f'logs/{logging_time}/task_{task}_window_len_{window_size}',
                f'model_{model_cls.__name__}/fold_{fold_idx}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Prepare data splits
            train_files = [data_files[i] for i in train_indices]
            test_files = [data_files[i] for i in test_indices]

            # Create datasets
            train_dataset = HARDataset(train_files, 
                                     window_size=window_size,
                                     overlap=overlap,
                                     dropped_channels=dropped_channels,
                                     logger=logger)
            test_dataset = HARDataset(test_files,
                                    window_size=window_size,
                                    overlap=overlap,
                                    dropped_channels=dropped_channels,
                                    logger=logger)
            
            # Normalize data
            train_mean, train_std = train_dataset.sample_mean, train_dataset.sample_std
            train_dataset.normalize_data(train_mean, train_std)
            test_dataset.normalize_data(train_mean, train_std)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Initialize model
            n_classes = 12 if task == 'activity' else 4
            
            try:
                model = model_cls(n_channels, n_classes)
            except TypeError:
                # Handle models with different initialization signatures
                if model_cls.__name__ == 'ResNet1D':
                    model = model_cls(BasicBlock1D, [3, 4, 6, 3], n_channels, n_classes)
                elif model_cls.__name__ == 'LSTM' and config_args:
                    model = model_cls(n_channels, n_classes, 
                                      hidden_size=config_args.lstm_hidden,
                                      num_layers=config_args.lstm_layers,
                                      bidirectional=config_args.bidirectional)
                else:
                    raise ValueError(f"Cannot instantiate model {model_cls.__name__}")
            
            model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training loop
            best_metric = 0
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                train_loss, train_accuracy = train_epoch(
                    model, train_loader, loss_fn, optimizer, device, task
                )
                
                # Validation phase
                valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate_epoch(
                    model, test_loader, loss_fn, device, task
                )
                
                # Store metrics
                results_df.loc[epoch] = [
                    epoch, train_loss, train_accuracy, valid_loss, 
                    valid_accuracy, valid_precision, valid_recall, valid_f1
                ]
                
                logger.info(f'Task: {task}, Window: {window_size}s, Fold: {fold_idx}, '
                           f'Model: {model_cls.__name__}, Epoch: {epoch}, '
                           f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                           f'Valid Acc: {valid_accuracy:.4f}')
                
                # Early stopping logic
                current_metric = results_df.at[epoch, mnt_metric]
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                
                if patience_counter > early_stop_patience:
                    logger.info(f"Early stopping triggered after {early_stop_patience} epochs")
                    logger.info(f"Best {mnt_metric}: {best_metric:.6f}")
                    break
            
            # Save results and cleanup
            results_df.to_csv(os.path.join(checkpoint_dir, 'results.csv'), index=False)
            best_metrics_per_fold.append(best_metric)
            
            del model, optimizer, train_loader, test_loader, train_dataset, test_dataset

        # Compute cross-validation statistics
        best_metric_list = np.array(best_metrics_per_fold)
        mean_metric = np.mean(best_metric_list)
        
        # Save CV results
        cv_results_path = os.path.join(
            log_path, f'logs/{logging_time}',
            f'task_{task}_window_{window_size}s_model_{model_cls.__name__}_best_metric_list.npy'
        )
        np.save(cv_results_path, best_metric_list)
        
        logger.info(f"=== Task: {task}, Window: {window_size}s, "
                   f"Model: {model_cls.__name__}, Mean CV {mnt_metric}: {mean_metric:.6f} ===")


if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser(description='Train deep learning models for HAR')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lstm_hidden', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--bidirectional', type=bool, default=True, help='Bidirectional LSTM')
    parser.add_argument('--data_path', type=str, default='data/dataset/',
                   help='Path to dataset directory')
    parser.add_argument('--log_base_dir', type=str, default='baseline/DL/',
                    help='Path to save logs and results')
    
    args = parser.parse_args()
    
    # Setup logging
    log_base_dir = args.log_base_dir
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger = setup_logger(os.path.join(log_base_dir, f'logs/{current_time}/logger.log'))
    
    # Save configuration
    config_path = os.path.join(log_base_dir, f'logs/{current_time}/config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    

    n_total_channels = N_SENSORS * N_CHANNELS_PER_SENSOR

    # Setup cross-validation
    kfold = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    cv_splits = list(kfold.split(np.arange(N_PARTICIPANTS)))
    
    # Dataset configuration
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run experiments for different window sizes
    window_sizes = [10, 7.5, 5, 2.5, 2, 1.5, 1, 0.5]  # seconds 
    
    for window_size in window_sizes:
        # Train ResNet1D-18, LSTM
        for model_cls in [ResNet1D, LSTM]:
            run_cross_validation(
                data_path=data_path,
                window_size=window_size,
                overlap=0.5,
                model_cls=model_cls,
                device=device,
                kf=cv_splits,
                tasks=['activity', 'intensity'],
                n_channels=n_total_channels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                logging_time=current_time,
                logger=logger
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
        
        # Convert to sensor indices
        body_part_to_index = {name: idx for idx, name in enumerate(BODY_PARTS)}
        dropped_sensor_indices = [body_part_to_index[name] for name in dropped_body_parts]
        
        # Convert to channel indices (each sensor has 6 channels)
        dropped_channels = []
        for sensor_idx in dropped_sensor_indices:
            sensor_channels = list(range(sensor_idx * N_CHANNELS_PER_SENSOR, 
                                    (sensor_idx + 1) * N_CHANNELS_PER_SENSOR))
            dropped_channels.extend(sensor_channels)

        n_remaining_channels = n_total_channels - len(dropped_channels)
        
        logger.info(f'Total channels: {n_total_channels}, '
                f'Remaining channels after dropping: {n_remaining_channels}')

        # Run experiments for a 2-second window
        window_sizes = [2]  # seconds
        
        for window_size in window_sizes:
            # Train ResNet1D-18
            for model_cls in [ResNet1D]:
                run_cross_validation(
                    data_path=data_path,
                    window_size=window_size,
                    overlap=0.5,
                    model_cls=model_cls,
                    device=device,
                    kf=cv_splits,
                    dropped_channels=dropped_channels,
                    tasks=['activity', 'intensity'],
                    n_channels=n_remaining_channels,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    logging_time=current_time,
                    logger=logger
                )