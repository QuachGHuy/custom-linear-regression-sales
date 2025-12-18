import os
import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load data from .csv
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    X = df.drop("Sales", axis=1).values
    Y = df["Sales"].values

    return X, Y

def split_data(X, Y, test_size=0.2):
    """
    Split_data
    
    X: Features data
    Y: Label data
    test_size: Size of data for testing
    """

    # 1. Get size
    data_size = X.shape[0]
    test_size = int(test_size * data_size)
    train_size = data_size - test_size

    # 2. Shuffle data
    np.random.seed(42) # <- Set random seed for the same result  
    indices = np.random.permutation(data_size)
    X = X[indices]
    Y = Y[indices]

    # 3. Split data
    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    return X_train, X_test, Y_train, Y_test

def data_scaler(X, mean=None, std=None):
    """
    Normalize for features data
    if mean/std is None: Normalize from X
    else: using for scaling
    """
    X = np.array(X)
    
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8 # <- Prevent devide zero
        
    X_scaled = (X - mean) / std

    return X_scaled, mean, std

def add_bias(X):
    """
    Add bias(w0) column for Features data
    """
    return np.c_[np.ones(X.shape[0]), X]
