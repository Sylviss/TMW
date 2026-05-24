"""
Dataset loading utilities for time series data.

This module provides utilities to load various time series datasets
in different formats (TXT, TSV, ARFF, etc.) with a unified interface.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings


def load_timeseries_from_txt(filepath: str) -> List[np.ndarray]:
    """
    Loads time series instances from a TXT file.
    
    Each instance consists of multiple lines of numbers.
    Instances are separated by one or more blank lines.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        List of NumPy arrays, each representing a time series instance
    """
    all_instances = []
    current_instance_lines = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                
                if not stripped_line:  # Found a blank line (separator)
                    if current_instance_lines:  # Process the instance we just read
                        try:
                            # Convert collected lines into a numpy array
                            instance_data = [
                                [float(num) for num in line_str.split()]
                                for line_str in current_instance_lines
                            ]
                            instance_array = np.array(instance_data, dtype=float)
                            all_instances.append(instance_array)
                        except ValueError as e:
                            print(f"Warning: Skipping instance due to non-numeric data. Error: {e}")
                        
                        # Reset for next instance
                        current_instance_lines = []
                else:
                    current_instance_lines.append(stripped_line)
            
            # Process the last instance if there's no trailing blank line
            if current_instance_lines:
                try:
                    instance_data = [
                        [float(num) for num in line_str.split()]
                        for line_str in current_instance_lines
                    ]
                    instance_array = np.array(instance_data, dtype=float)
                    all_instances.append(instance_array)
                except ValueError as e:
                    print(f"Warning: Skipping final instance due to non-numeric data. Error: {e}")
                    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return []
    
    return all_instances


def load_ucr_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCR time series dataset format.
    
    UCR format: first column is label, remaining columns are time series values.
    
    Args:
        filepath: Path to the UCR dataset file
        
    Returns:
        Tuple of (data, labels) as NumPy arrays
    """
    try:
        data = np.loadtxt(filepath)
        labels = data[:, 0].astype(int)
        time_series = data[:, 1:]
        return time_series, labels
    except Exception as e:
        print(f"Error loading UCR dataset from '{filepath}': {e}")
        return np.array([]), np.array([])


def load_tsv_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TSV (Tab-Separated Values) dataset.
    
    Args:
        filepath: Path to the TSV file
        
    Returns:
        Tuple of (data, labels) as NumPy arrays
    """
    try:
        df = pd.read_csv(filepath, sep='\t', header=None)
        labels = df.iloc[:, 0].values.astype(int)
        data = df.iloc[:, 1:].values.astype(float)
        return data, labels
    except Exception as e:
        print(f"Error loading TSV dataset from '{filepath}': {e}")
        return np.array([]), np.array([])


def load_arff_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ARFF (Attribute-Relation File Format) dataset.
    
    Note: This is a simplified ARFF loader for time series data.
    
    Args:
        filepath: Path to the ARFF file
        
    Returns:
        Tuple of (data, labels) as NumPy arrays
    """
    data_section = False
    data_list = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith('@data'):
                    data_section = True
                    continue
                
                if data_section and line and not line.startswith('%'):
                    # Parse data line
                    values = line.split(',')
                    # Last value is typically the class label
                    data_values = [float(v) for v in values[:-1]]
                    label = values[-1].strip().strip("'\"")
                    data_list.append(data_values + [label])
        
        if data_list:
            data_array = np.array(data_list, dtype=object)
            time_series = data_array[:, :-1].astype(float)
            labels = data_array[:, -1]
            return time_series, labels
        else:
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"Error loading ARFF dataset from '{filepath}': {e}")
        return np.array([]), np.array([])


class DatasetLoader:
    """
    Unified dataset loader for various time series datasets.
    """
    
    def __init__(self, data_root: str = "consolidated_datasets"):
        """
        Initialize the dataset loader.
        
        Args:
            data_root: Root directory containing all datasets
        """
        self.data_root = Path(data_root)
        
    def load_dataset(self, dataset_name: str, split: str = "TRAIN") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific dataset by name.
        
        Args:
            dataset_name: Name of the dataset (folder name)
            split: Either "TRAIN" or "TEST"
            
        Returns:
            Tuple of (data, labels) as NumPy arrays
        """
        dataset_path = self.data_root / dataset_name
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found in {self.data_root}")
        
        # Try different file formats
        for ext in ['.txt', '.tsv', '.arff']:
            filepath = dataset_path / f"{dataset_name}_{split}{ext}"
            if filepath.exists():
                if ext == '.txt':
                    return load_ucr_dataset(str(filepath))
                elif ext == '.tsv':
                    return load_tsv_dataset(str(filepath))
                elif ext == '.arff':
                    return load_arff_dataset(str(filepath))
        
        raise FileNotFoundError(f"No suitable file found for dataset '{dataset_name}' split '{split}'")
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        if not self.data_root.exists():
            return []
        
        datasets = []
        for item in self.data_root.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        try:
            train_data, train_labels = self.load_dataset(dataset_name, "TRAIN")
            test_data, test_labels = self.load_dataset(dataset_name, "TEST")
            
            info = {
                'name': dataset_name,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'sequence_length': train_data.shape[1] if len(train_data) > 0 else 0,
                'num_classes': len(np.unique(train_labels)) if len(train_labels) > 0 else 0,
                'classes': np.unique(np.concatenate([train_labels, test_labels])).tolist()
            }
            return info
            
        except Exception as e:
            return {'name': dataset_name, 'error': str(e)}


def create_synthetic_time_series(length: int, dimension: int = 1, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic time series data for testing.
    
    Args:
        length: Length of the time series
        dimension: Number of dimensions/features
        noise_level: Amount of noise to add
        
    Returns:
        NumPy array of shape (length, dimension)
    """
    t = np.linspace(0, 4*np.pi, length)
    
    if dimension == 1:
        # Simple sine wave with noise
        series = np.sin(t) + noise_level * np.random.randn(length)
        return series.reshape(-1, 1)
    elif dimension == 2:
        # 2D spiral pattern
        series = np.column_stack([
            np.sin(t) * np.exp(-t/10) + noise_level * np.random.randn(length),
            np.cos(t) * np.exp(-t/10) + noise_level * np.random.randn(length)
        ])
        return series
    else:
        # Multi-dimensional random walk
        return np.cumsum(np.random.randn(length, dimension), axis=0)


def create_2d_discrete_distribution(mu: float = 0.4, sigma: float = 0.05, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D discrete distribution with uniform x and normal-like y.
    
    Args:
        mu: Mean of the Gaussian
        sigma: Standard deviation of the Gaussian
        num_points: Number of discrete points
        
    Returns:
        Tuple of (x_values, y_probabilities)
    """
    # Uniformly distributed x values from 0 to 1
    x_values = np.linspace(0, 1, num_points)
    
    # Gaussian function
    y_probabilities = np.exp(-((x_values - mu) ** 2) / (2 * sigma ** 2))
    
    return x_values, y_probabilities
