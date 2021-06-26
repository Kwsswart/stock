import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas_datareader as pdr


# Import AAPL stock price
df_aapl = pdr.get_data_yahoo('AAPL', start="2019-01-01", end="2019-09-30")

# import SPY stock price
df_spy = pdr.get_data_yahoo("SPY", start="2019-01-01", end="2019-09-30")

df_aapl[["Open", "High", "Low", "Close"]].plot()
plt.show()
df_aapl[["Open", "High", "Low", "Close"]].plot()
plt.show()


fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()

plot_data=[]
for i in range(150, len(df_aapl)):
    # Formatting the data under the OHLC format "Open-High-Low-Close"
    """
    Open: is the price of a stock when a time resolution started (1m, 30m, hourly, daily, etc)
    High: is the highest price reached from the beginning to the end of candle.
    Low: is the lowest price reached from the beginning to the end of candle.
    Close: is the price of a stock when a time resolution finishes.
    """
    row = [
        i,
        df_aapl.Open.iloc[i],
        df_aapl.High.iloc[i],
        df_aapl.Low.iloc[i],
        df_aapl.Close.iloc[i],
    ]
    plot_data.append(row)
candlestick_ohlc(ax, plot_data)
plt.show()


# From the stockstats library we will import StockDataFrame, which is a class that receives as an attribute a pandas DataFrame
# sorted by time and includes the columns Open-Close-High-Low in this order:


"""
Using this we can see the trends better to know whether they are upwards or downwards
"""

from stockstats import StockDataFrame

stocks = StockDataFrame.retype(df_aapl[["Open", "Close", "High", "Low", "Volume"]])

# Simple Moving Average is an indicator which smoothens the stock prices plot, by computing the mean of the prices over a period of time.
# This will allow us to better visualize the trends directions (up or down).


# Plotting SMA:

plt.plot(stocks["close_10_sma"], color="b", label="SMA")
plt.plot(df_aapl.Close, color="g", label="Close prices")
plt.legend(loc="lower right")
plt.show()


"""
SMA PlotTo compute the SMA, we specify the attribute under the following format:

Column_Period_Indicator

In our case, we are concerned about the Close prices within a period of 10 and
the indicator is SMA, so the result is close_10_sma.
"""


"""Expontential Moving Average (EMA)"""

'''
Unlike the SMA, which gives equal weights to all prices whether the old or new ones, 
the Exponential Moving Average emphasizes on the last (most recent) prices by attributing to them a greater weight, 
this makes the EMA an indicator which detects trends faster than the SMA.
'''

# Formula is the following:
# WeightedMultiplier = 2/(N+1)
# EMA(P) = (Close(P) - EMA(P - 1) * WeightedMultiplier + EMA(P - 1))

# Plotting:

plt.plot(stocks["close_10_sma"], color="b", label="SMA") # Plotting SMA
plt.plot(stocks["close_10_ema"], color="k", label="EMA")
plt.plot(df_aapl.Close, color="g", label="Close prices") # Plotting Closing prices
plt.legend(loc="lower right")
plt.show()

# In the zoom figure, we can clearly observe that indeed EMA responds faster to the change of trends and gets closer to them.


""" Moving Average Convergence/Divergence (MACD) """

'''
Moving Average Convergence/Divergence is a trend-following momentum indicator. 
This Indicator can show changes in the speed of price movement and traders use it to determine the direction of a trend. 

The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. 
A nine-day EMA of the MACD is called “Signal Line”, which is then plotted with the MACD. 

When MACD crosses below the Signal Line it is an indicator to start doing short (Sell) operations.
And when it crosses above it, it is an indicator to start doing long (Buy) operations:
'''

plt.plot(stocks["macd"], color="b", label="MACD")
plt.plot(stocks["macds"], color="g", label="Signal Line")
plt.legend(loc="lower right")
plt.show()