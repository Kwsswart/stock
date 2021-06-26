import numpy as np

import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    """
    This function plots true close price (blue) along with the predicted close price (red)
    :param test_df:
    :return:
    """

    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    """
    This takes the model and the data dict to construct a final dataframe that includes the features
    along with true and predicted prices of the testing dataset.
    :param model:
    :param data:
    :return:
    """

    # if the predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit.
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0

    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    test_df = data["test_df"]

    # Add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred

    # Add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test

    # Sort data by date
    test_df.sort_index(inplace=True)

    final_df = test_df

    # add the buy profit column

    final_df["buy_profit"] = list(map(
        buy_profit,
        final_df["adjclose"],
        final_df[f"adjclose_{LOOKUP_STEP}"],
        final_df[f"true_adjclose_{LOOKUP_STEP}"]
        # Since no profit from last sequence, add 0`s
    ))

    # add the sell profit column
    final_df["sell_profit"] = list(map(
        sell_profit,
        final_df["adjclose"],
        final_df[f"adjclose_{LOOKUP_STEP}"],
        final_df[f"true_adjclose_{LOOKUP_STEP}"])
        # since we don't have profit for last sequence, add 0's
    )
    return final_df


def predict(model, data):
    # Retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]

    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # get the prediction (scaling from 0 to 1)
    prediction = model.predict(last_sequence)

    # get the price by inverting the scale
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]

    return predicted_price


def main():
    # load the data
    data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                     shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS)

    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS,
                         units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # loading optimal model weights from results
    model_path = os.path.join(os.getcwd(), "results", model_name) + '.h5'
    model.load_weights(model_path)

    # Here I calculate the loss and mean absolute error using model.evaluate():
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)

    # calculate mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # taking scaled output values into consideration, so we use the inverse_transform() method from the MinMaxScaler we
    # defined in the load_data() function earlier if SCALE parameter was set to True.

    final_df = get_final_df(model, data)

    # predict future price
    future_price = predict(model, data)

    # calculate the accuracy score by counting the number of positive profits in both buy profit and sell profit
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)

    # total buy and sell profit
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df['sell_profit'].sum()

    total_profit = total_sell_profit + total_buy_profit
    # divide this by the number of test samples

    profit_per_trade = total_profit / len(final_df)

    # printing metrics
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    # plot true/pred prices graph
    plot_graph(final_df)
    print(final_df.tail(10))

    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)


if __name__ == "__main__":
    main()