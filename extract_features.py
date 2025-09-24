"""
Feature extraction module for human activity recognition data.

This module processes raw sensor data from 17 body-worn sensors and 
extracts time-domain and frequency-domain features using sliding windows 
of eight lengths (0.5~10 s) with 50% overlap.
"""

import os
import pandas as pd
import numpy as np
import tqdm
from scipy.stats import entropy
from argparse import ArgumentParser


# Constants
N_PARTICIPANTS = 25
N_SENSORS = 17
SAMPLE_RATE = 60  # Hz
AXES = ['X', 'Y', 'Z']
SENSOR_TYPE = ['Acc', 'Gyr']
BODY_PARTS = [
    'LowerBack', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 
    'LeftShank', 'LeftFoot', 'UpperBack', 'Head', 'RightShoulder', 
    'RightUpperArm', 'RightForeArm', 'RightWrist', 'LeftShoulder', 
    'LeftUpperArm', 'LeftForeArm', 'LeftWrist'
]


def signal_entropy(signal, bins=10):
    """
    Calculate signal entropy using histogram-based probability estimation.
    
    Args:
        signal: Input signal array
        bins: Number of histogram bins
        
    Returns:
        float: Entropy value of the signal
    """
    try:
        hist, _ = np.histogram(signal, bins=bins, density=True)
        return entropy(hist)
    except:
        return np.nan


class FeatureExtractor:
    """Extract time-domain and frequency-domain features from sensor signals."""
    
    def __init__(self, sample_rate):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sampling frequency in Hz
        """
        self.sample_rate = sample_rate
        self.cache = {}

    def clear_cache(self):
        """Clear frequency domain computation cache."""
        self.cache = {}

    def compute_psd(self, signal, key):
        """
        Compute power spectral density using FFT.
        
        Args:
            signal: Input signal array
            key: Cache key for storing computation results
            
        Returns:
            tuple: Frequencies, PSD values, and normalized probability distribution
        """
        if key not in self.cache:
            N = len(signal)
            X = np.fft.fft(signal)
            freqs = np.fft.fftfreq(N, d=1/self.sample_rate)
            
            # Single-sided spectrum
            half_N = int(np.floor(N / 2) + 1)
            freqs_ss = freqs[:half_N]
            
            # Power spectral density
            PSD = (np.abs(X)**2) / N
            PSD_ss = PSD[:half_N]
            PSD_ss[1:] = 2 * PSD_ss[1:]  # Double amplitude for non-DC components
            p = PSD_ss / np.sum(PSD_ss + 1e-10)  # Normalize to probability distribution
            
            self.cache[key] = {
                'freqs': freqs_ss,
                'psd': PSD_ss,
                'p': p
            }
        return self.cache[key]['freqs'], self.cache[key]['psd'], self.cache[key]['p']

    # Time-domain features
    def mean(self, x):
        """Calculate signal mean."""
        return np.mean(x)
        
    def var(self, x):
        """Calculate signal variance."""
        return np.var(x)
        
    def max(self, x):
        """Calculate signal maximum."""
        return np.max(x)
        
    def min(self, x):
        """Calculate signal minimum."""
        return np.min(x)
        
    def range(self, x):
        """Calculate signal range (max - min)."""
        return np.max(x) - np.min(x)
        
    def skewness(self, x):
        """Calculate signal skewness."""
        return pd.Series(x).skew()
        
    def energy(self, x):
        """Calculate signal energy (sum of squares)."""
        return np.sum(np.square(x))
        
    def entropy(self, x):
        """Calculate signal entropy."""
        return signal_entropy(x + 1e-10)
        
    def iqr(self, x):
        """Calculate interquartile range."""
        return np.percentile(x, 75) - np.percentile(x, 25)
        
    def mad(self, x):
        """Calculate mean absolute deviation."""
        return np.mean(np.abs(x - np.mean(x)))
        
    def rms(self, x):
        """Calculate root mean square."""
        return np.sqrt(np.mean(np.square(x)))
        
    def sma(self, x):
        """Calculate signal magnitude area."""
        return np.mean(np.abs(x))
        
    def zero_crossing_rate(self, x):
        """Calculate zero crossing rate."""
        return len(np.where(np.diff(np.sign(x)))[0]) / len(x)
        
    def mean_crossing_rate(self, x):
        """Calculate mean crossing rate."""
        return len(np.where(np.diff(np.sign(x - np.mean(x))))[0]) / len(x)

    # Frequency-domain features
    def spectral_centroid(self, x, key):
        """Calculate spectral centroid."""
        freqs, _, p = self.compute_psd(x, key)
        return np.sum(freqs * p)

    def spectral_variance(self, x, key):
        """Calculate spectral variance."""
        freqs, _, p = self.compute_psd(x, key)
        mean_freq = np.sum(freqs * p)
        return np.sum((freqs - mean_freq) ** 2 * p)

    def spectral_entropy(self, x, key):
        """Calculate spectral entropy."""
        _, _, p = self.compute_psd(x, key)
        return -np.sum(p * np.log(p + 1e-10))

    def dominant_frequency(self, x, key):
        """Calculate dominant frequency (peak frequency)."""
        freqs, psd, _ = self.compute_psd(x, key)
        return freqs[np.argmax(psd)]


def extract_features(data_path, feature_path, window_size, overlap=0.5):
    """
    Extract features from sensor data using sliding windows.
    
    Args:
        data_path: Path to raw data directory
        feature_path: Path to save extracted features
        window_size: Window size in seconds
        overlap: Overlap ratio between consecutive windows
    """
    # Validate path to raw data directory
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    
    # Create output directories
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
        
    window_dir = os.path.join(feature_path, f'{window_size}s')
    if not os.path.exists(window_dir):
        os.makedirs(window_dir)

    # Define feature list
    feature_names = [
        'mean', 'var', 'max', 'min', 'range', 'skewness', 
        'energy', 'entropy', 'iqr', 'mad', 'rms', 'sma',
        'zero_crossing_rate', 'mean_crossing_rate',
        'spectral_centroid', 'spectral_variance', 'spectral_entropy', 'dominant_frequency'
    ]
    
    # Generate feature column names
    feature_columns = ['Activity'] + [
        f'{sensor_type}_{axis}_{BODY_PARTS[sensor_idx]}_{feature_name}'
        for sensor_idx in range(N_SENSORS) 
        for sensor_type in SENSOR_TYPE 
        for axis in AXES
        for feature_name in feature_names
    ]

    # Window parameters
    window_samples = int(window_size * SAMPLE_RATE) # Number of data samples in each sliding window
    overlap_samples = int(window_samples * overlap) # Overlap between consecutive windows in samples
    step_samples = int(window_samples * (1 - overlap)) # Step in samples

    extractor = FeatureExtractor(SAMPLE_RATE)

    # Process each participant
    print(f"Processing {window_size}s windows...")
    for participant_id in range(N_PARTICIPANTS):  
        file_path = os.path.join(data_path, f'P{participant_id+1:02d}.csv')
        data = pd.read_csv(file_path)
        output_path = os.path.join(window_dir, f'features_P{participant_id+1:02d}.csv')
        if os.path.exists(output_path):
            print(f"Skipping participant {participant_id+1} as features already exist.")
            continue

        # Extract features using sliding windows
        features_df = pd.DataFrame(columns=feature_columns)
        
        n_timesteps = len(data)
        for i in tqdm.tqdm(range(0, n_timesteps - window_samples + 1, step_samples)):
            window_data = data.iloc[i:i + window_samples]
            activity_labels = np.unique(window_data['Activity'].values)
            
            # Skip windows with multiple activities or invalid labels (0)
            if len(activity_labels) > 1 or activity_labels[0] == 0:
                continue
                
            window_data = window_data.drop(columns=['Activity'])
            
            feature_row = [activity_labels[0]]
            extractor.clear_cache()
            
            # Extract features for each sensor channel
            for feature_col in feature_columns[1:]:
                parts = feature_col.split('_')
                sensor_col = f"{parts[0]}_{parts[1]}_{parts[2]}"
                feature_name = feature_col.split('_',3)[-1]
                
                signal = window_data[sensor_col].values
                method = getattr(extractor, feature_name)
                
                # Apply frequency-domain or time-domain feature extraction
                if 'spectral' in feature_name or feature_name == 'dominant_frequency':
                    feature_value = method(signal, sensor_col)
                else:
                    feature_value = method(signal)
                    
                feature_row.append(feature_value)
            
            # Validate feature values
            assert np.inf not in feature_row, f"Inf value found in features"
            assert -np.inf not in feature_row, f"-Inf value found in features"
            assert not any(pd.isna(feature_row)), f"NaN value found in features"
            
            features_df.loc[len(features_df)] = feature_row
        
        # Save features for current participant
        features_df.to_csv(output_path, index=False)
        print(f"Features for participant {participant_id+1} saved successfully!")

    # Merge features from all participants
    all_features = pd.DataFrame(columns=feature_columns)
    
    for participant_id in range(N_PARTICIPANTS):
        feature_file = os.path.join(window_dir, f'features_P{participant_id+1:02d}.csv')
        
        if not os.path.exists(feature_file):
            print(f"File {feature_file} does not exist.")
            continue
            
        participant_features = pd.read_csv(feature_file)
        all_features = pd.concat([all_features, participant_features], ignore_index=True)
        
    # Save merged features
    merged_path = os.path.join(window_dir, 'features_all.csv')
    all_features.to_csv(merged_path, index=False)
    print(f'All features saved to {merged_path}')


if __name__ == '__main__':
    # Configuration
    parser = ArgumentParser(description='Train deep learning models for HAR')
    parser.add_argument('--data_path', type=str, default='data/dataset/',
                   help='Path to raw data directory')
    parser.add_argument('--feature_path', type=str, default='data/features/',
                   help='Path to save extracted features')
    args = parser.parse_args()

    data_path = args.data_path
    feature_path = args.feature_path
    window_sizes = [10, 7.5, 5, 2.5, 2, 1.5, 1, 0.5]  # seconds
    
    # Extract features for different window sizes
    for window_size in window_sizes:
        extract_features(data_path, feature_path, window_size)