import os
import time
from tensorflow.keras.layers import LSTM


# Window size / sequence length
N_STEPS = 50

# lookup step, 1 is the next day
LOOKUP_STEP = 15

# Scale feature columns & output price
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# Whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# Whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{SHUFFLE}"

# test ratio size, 0.2 = 20%
TEST_SIZE = 0.2

# features used
FEATURE_COLUMNS = [
    "adjclose",
    "volume",
    "open",
    "high",
    "low"
]

# date now
date_now = time.strftime("%Y-%m-%d")


"""Model Params"""

N_LAYERS = 2

# LSTM CELL
CELL = LSTM

# 256 LSTM NEURONS
UNITS = 256

# 40% DROPOUT
DROPOUT = 0.4

# WHETHER TO USE BIDIRECTIONAL RNNS
BIDIRECTIONAL = False



""" Training Params"""

# mean absolute error loss
# LOSS = "mae"
# huber loss

LOSS = "huber_loss"

OPTIMIZER = "adam"

BATCH_SIZE = 64

# EPOCHS = 500

EPOCHS = 2000
# Amazon stock market
ticker = "AMZN"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"