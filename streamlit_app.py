import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D
from sklearn.metrics import r2_score
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Stock Predictor", page_icon=":chart_with_upwards_trend:")

@st.cache
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    return data

def decision_tree_app():
    st.title("Decision Tree Regression")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-05-03"))

    df = fetch_stock_data(ticker, start, end)

    st.subheader("Data Summary")
    st.write(df.head())
    st.write(df.describe())

    st.subheader("Data Visualization")
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Stock Prices Over Time")
    axs[0, 0].plot(df['Open'])
    axs[0, 0].set_title("Opening Price")
    axs[0, 1].plot(df['High'])
    axs[0, 1].set_title("High Price")
    axs[1, 0].plot(df['Low'])
    axs[1, 0].set_title("Low Price")
    axs[1, 1].plot(df['Close'])
    axs[1, 1].set_title("Closing Price")
    st.pyplot(fig)

    df2 = pd.DataFrame(df['Close'])
    df2['Prediction'] = df2['Close'].shift(-100)
    X = np.array(df2.drop(['Prediction'], axis=1))[:-100]
    y = np.array(df2['Prediction'])[:-100]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree = DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)

    future_days = 100
    x_future = df2.drop(['Prediction'], axis=1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    tree_prediction = tree.predict(x_future)
    lr_prediction = lr.predict(x_future)

    st.subheader("Predictions")
    st.write("Decision Tree Regression Prediction:")
    st.write(tree_prediction)
    st.write("Linear Regression Prediction:")
    st.write(lr_prediction)

    predictions = tree_prediction
    valid = df2[X.shape[0]:]
    valid['Predictions'] = predictions
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title("Stock Prices Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price USD ($)")
    ax.plot(df2['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(["Actual Close Price", "Predicted Close Price (Decision Tree Regression)", "Predicted Close Price (Linear Regression)"])
    st.pyplot(fig)

def lstm_app():
    st.title("LSTM (Long Short-Term Memory)")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-05-03"))
    df = fetch_stock_data(ticker, start, end)
    st.subheader("Data Summary")
    st.write(df.head())
    st.write(df.describe())

    st.subheader("Data Visualization")
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Stock Prices Over Time")
    axs[0, 0].plot(df['Open'])
    axs[0, 0].set_title("Opening Price")
    axs[0, 1].plot(df['High'])
    axs[0, 1].set_title("High Price")
    axs[1, 0].plot(df['Low'])
    axs[1, 0].set_title("Low Price")
    axs[1, 1].plot(df['Close'])
    axs[1, 1].set_title("Closing Price")
    st.pyplot(fig)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    prediction_days = 100
    x_train = scaled_data[:-prediction_days]
    y_train = scaled_data[prediction_days:]
    x_test = scaled_data[-prediction_days:]

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_data = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    predictions = model.predict(test_data)
    predictions = scaler.inverse_transform(predictions)

    st.subheader("Predictions")
    st.write(predictions)

    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title("Stock Prices Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price USD ($)")
    ax.plot(df['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(["Actual Close Price", "Predicted Close Price (LSTM)"])
    st.pyplot(fig)

def main():
    st.title("Stock Price Prediction App")
    st.sidebar.title("Select Model")
    model = st.sidebar.selectbox("Choose a model", ("Decision Tree Regression", "LSTM (Long Short-Term Memory)"))
    if model == "Decision Tree Regression":
        decision_tree_app()
    elif model == "LSTM (Long Short-Term Memory)":
        lstm_app()
    if name == "main":
        main()
