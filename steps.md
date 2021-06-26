# stock_prediction:

## load_data:

This function is long but handy, it accepts several arguments to be as flexible as possible:

- The **ticker** argument is the ticker we want to load, for instance, you can use TSLA for Tesla stock market, AAPL for Apple and so on. It can also be a pandas Dataframe with the condition it includes the columns in feature_columns as well as date as index.
- **n_steps** integer indicates the historical sequence length we want to use, some people call it the window size, recall that we are going to use a recurrent neural network, we need to feed in to the network a sequence data, choosing 50 means that we will use 50 days of stock prices to predict the next lookup time step.
- **scale** is a boolean variable that indicates whether to scale prices from 0 to 1, we will set this to True as scaling high values from 0 to 1 will help the neural network to learn much faster and more effectively.
- **lookup_step** is the future lookup step to predict, the default is set to 1 (e.g next day). 15 means the next 15 days, and so on.
- **split_by_date** is a boolean which indicates whether we split our training and testing sets by date, setting it to False means we randomly split the data into training and testing using sklearn's train_test_split() function. If it's True (the default), we split the data in date order.
- We will be using all the features available in this dataset, which are the open, high, low, volume and adjusted close. Please check this tutorial to learn more what these indicators are.

The above function does the following:

- First, it loads the dataset using stock_info.get_data() function in yahoo_fin module.
- It adds the "date" column from the index if it doesn't exist, this will help us later to get the features of the testing set.
- If the scale argument is passed as True, it will scale all the prices from 0 to 1 (including the volume) using the sklearn's MinMaxScaler class. Note that each column has its own scaler.
- It then adds the future column which indicates the target values (the labels to predict, or the y's) by shifting the adjusted close column by lookup_step.
- After that, it shuffles and splits the data to training and testing sets, and finally returns the result.

To understand the code even better, I highly suggest you to manually print the output variable (result) and see how the features and labels are made.


## create_model:

This function creates a recursive neural network (RNN) that has a dense layer as output layer with 1 neuron, this model requires a sequence of features of sequence_length (i.e. 50 or 100) consecutive time steps (days with this dataset) and outputs a single value which indicates the price of the next time step.

This accepts n_features as an argument, which is the number of features we will pass on each sequence, this case: *adjclose, open, high, low, volume (5 features)*

- **n_layers** is the number of RNN layers we want to stack.
- **dropout** is the dropout rate after each RNN layer.
- **units are** the number of RNN cell units (Whether [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM), [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN) or [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU))
- **bidirectional** is a boolean that indicates whether to use [bidirectional RNNs](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks)



# parameters.py

- **TEST_SIZE** - The testing set rate. I.e: 0.2 means 20% of the total dataset.
- **FEATURE_COLUMNS** - The features we will use to predict the next price value.
- **N_LAYERS** - Number of RNN layers to use.
- **CELL** - RNN cell to use Default is LSTM.
- **UNITS** - Number of cell units.
- **DROPOUT** - This is the dropout rate or the probability of not training a given node in a layer, where 0.0 means no dropout at all. This regularization can help the model to not overfit on the training data.
- **BIDIRECTIONAL** - Whether to use [bidirectional recurrent neural networks](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks)
- **LOSS** - Loss function to use for this regression problem, here it is [Huber loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber), calso can use mean absolute error (mae) or mean squared error (mse).
- **OPTIMIZER** - Optimization algorithm used def = [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
- **BATCH_SIZE** - The number of data samples to use on each training iteration.
- **EPOCHS** - The number of times that the learning algorithm will pass through the entire training data set, 500 is the minimum try to increase more



# Training

Once running the training it will take a while, increase the number of Epochs for better results.
After the training ends, try to run tensorboard with:
- tensorboard --logdir="logs"

Now this will start a local HTTP server at localhost:6006, after going to the browser we can find a graph.

Once trained the model we can evaluate it and see how it's doing on the testing set