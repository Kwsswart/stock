from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import pandas as pd
from parameters import *

# Ensure folders are created if not existing
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")


# Train model
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_step=LOOKUP_STEP,
                 test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

# Save the dataframe
data["df"].to_csv(ticker_data_filename)

# build model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# Tensoreflow callbacks.
## Setting checkpoints.
# This one saves the model in each epoch during the training.
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True,
                               verbose=1)
## Setting log
# This is to visualize the model performance.
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# Train the model and retain the weights whenever we see a new optimal model.

history = model.fit(
    data["X_train"],
    data["y_train"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(data["X_test"], data["y_test"]),
    callbacks=[checkpointer, tensorboard],
    verbose=1
)