import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Stock Predictor", page_icon=":chart_with_upwards_trend:")

st.title("Stock Predictor")

# Define stock ticker input
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")

# Download the data
start = '2010-01-01'
end = '2019-12-31'
df = yf.download(ticker, start, end)

# Show data summary
st.subheader("Data Summary")
st.write(df.head())
st.write(df.describe())

# Visualize the data
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

# Prepare data for prediction
df2 = pd.DataFrame(df['Close'])
df2['Prediction'] = df2['Close'].shift(-100)
X = np.array(df2.drop(['Prediction'], axis=1))[:-100]
y = np.array(df2['Prediction'])[:-100]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

# Predict future prices
future_days = 100
x_future = df2.drop(['Prediction'], axis=1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)

# Show predictions
st.subheader("Predictions")
st.write("Decision Tree Regression Prediction:")
st.write(tree_prediction)
st.write("Linear Regression Prediction:")
st.write(lr_prediction)

# Visualize predictions
predictions = tree_prediction
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("Stock Prices Over Time")
ax.set_xlabel("Days")
ax.set_ylabel("Closing Price USD ($)")
ax.plot(df2['Close'])
ax.plot(valid[['Close', 'Predictions']])
ax.legend(["Original", "Valid", "Predicted"])
st.pyplot(fig)
