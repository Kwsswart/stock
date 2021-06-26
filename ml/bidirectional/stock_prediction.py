import os
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Look into using https://github.com/RomelTorres/alpha_vantage instead of yahoo
from yahoo_fin import stock_info as si

from collections import deque


# Setting seeds, to ensure that results are same after rerunning

np.random.seed(314)
# tf.random.set_random_seed(314)
tf.random.set_seed(314)
random.seed(314)

# preparing dataset


def shuffle_in_unison(a, b):
    """
    Function designed to shuffle two arrays in the same way
    :param a:
    :param b:
    :return:
    """

    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_data(ticker, n_steps=50, scale=True,
              shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=["adjclose", "volume", "open", "high", "low"]):
    """
    Function to load data from Yahoo funance soure, as well as handle scaling, shuffling, normalizing and splitting.
    :param ticker: (str/pd.DF) - The ticker we want to load, i.e. AAPL, TESL, etc.
    :param n_steps: (int) - The historical sequence length (i.e window size) used to predict. Def=50
    :param scale: (bool) - Whether to scale prices from 0 to 1. Def = True
    :param shuffle: (bool) - Whether to shuffle the dataset (both training % testing). Def = True
    :param lookup_step: (int) - The future lookup step to predict. Def = 1 (Next day)
    :param split_by_date: (bool) - Whether we split the dataset iinto training/testing by date, setting it to False will randomly split datasets.
    :param test_size: (float) - Ratio for test data. Def = 0.2 (20% Testing data)
    :param feature_columns: (list) - The list of features to use to feed into the model, Def= Everything taken from yahoo_fin
    :return:
    """

    # Check if ticker is already a loaded stock from yahoo finance

    if isinstance(ticker, str):
        # load it from the yahoo library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # Already loaded, thus can use directly
        df = ticker
    else:
        raise TypeError("ticker can be either str or a pd.DataFrame")

    result = dict()

    result["df"] = df.copy() # Setting original df within result

    for col in feature_columns:
        # Check that all columns exist within dataframe
        assert col in df.columns, f"{col} does not exist in dataframe."

    # Add date
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = dict()

        # Scale the prices from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['adjclose'].shift(-lookup_step)

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))

        # drop NaNs
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=n_steps)

        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # Get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)

        # add to result
        result['last_sequence'] = last_sequence

        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        if split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)

            train_samples = int((1 - test_size) * len(X))

            # Train variables
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]

            # Test variables
            result["X_test"] = X[train_samples:]
            result["y_test"] = y[train_samples:]

            if shuffle:
                # shuffle the datasets for training (if shuffle parameter is set)
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        else:
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                        test_size=test_size,
                                                                                                        shuffle=shuffle)

        # Get the list of test set dates
        dates = result["X_test"][:, -1, -1]

        # Retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]

        # remove duplicated dates in the testing dataframe while inverting the bits
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

        return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    '''
    Function to create the sequencial model
    AThis function is flexible, the following is changable:
        - the number of layers,
        - dropout rate,
        - the RNN cell,
        - loss
        - the optimizer used to compile the model.
    :param sequence_length:
    :param n_features:
    :param units:
    :param cell:
    :param n_layers:
    :param dropout:
    :param loss:
    :param optimizer:
    :param bidirectional:
    :return:
    '''

    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(
                    Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features))
                )
            else:
                model.add(
                    cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features))
                )
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

        # Add dropout after each layer
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model