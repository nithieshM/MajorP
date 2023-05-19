# Import necessary libraries
import streamlit as st
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title='Stock Price Forecasting with SVM')

# Define function to fetch data from Yahoo Finance
@st.cache
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    data = data[['Open-Close', 'High-Low', 'Close']]
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna()
    return data

# Define function to split data into train and test sets
def split_data(data, split_percentage):
    split = int(split_percentage * len(data))
    X_train = data[:split][['Open-Close', 'High-Low']]
    y_train = data[:split]['Target']
    X_test = data[split:][['Open-Close', 'High-Low']]
    y_test = data[split:]['Target']
    return X_train, y_train, X_test, y_test

# Define function to train SVM model and generate predictions
def train_model(X_train, y_train, X_test):
    clf = SVC().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

# Define function to plot the results
def plot_results(data):
    data['Return'] = data.Close.pct_change()
    data['Strategy_Return'] = data.Return * data.Predicted_Signal.shift(1)
    data['Cum_Ret'] = data['Return'].cumsum()
    data['Cum_Strategy'] = data['Strategy_Return'].cumsum()

    plt.plot(data['Cum_Ret'], color='red', label='Buy and Hold')
    plt.plot(data['Cum_Strategy'], color='blue', label='SVM Strategy')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    st.pyplot()

# Define Streamlit app
def main():
    st.title('Stock Price Forecasting with SVM')
    st.write('Enter the stock ticker below to see the SVM-based stock price forecasting and visualization.')

    # Get user inputs
    ticker = st.text_input('Stock Ticker', 'AAPL')
    start = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
    end = st.date_input('End Date', value=pd.to_datetime('2022-05-03'))
    split_percentage = st.slider('Training-Testing Set Split Percentage', 0.1, 0.9, 0.8, 0.1)

    # Fetch and split data
    data = fetch_data(ticker, start, end)
    X_train, y_train, X_test, y_test = split_data(data, split_percentage)

    # Train model and generate predictions
    y_pred = train_model(X_train, y_train, X_test)
    data['Predicted_Signal'] = np.concatenate((np.zeros(len(y_train)), y_pred))

    # Plot results
    plot_results(data)

# Run Streamlit app
if __name__ == '__main__':
    main()
