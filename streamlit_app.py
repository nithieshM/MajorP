'''import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D
from sklearn.metrics import r2_score
from sklearn import metrics

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set up the page title

st.set_page_config(page_title='Stock Price Forecasting', page_icon=':chart_with_upwards_trend:')

# Set up the sidebar
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Enter a stock symbol:', 'AAPL')
start_date = st.sidebar.date_input('Start date:', value=pd.to_datetime('2009-01-01'))
end_date = st.sidebar.date_input('End date:', value=pd.to_datetime('2022-12-31'))

# Download the stock data
data = yf.download(ticker, start_date, end_date)
df = pd.DataFrame(data)
df.dropna(inplace=True)

# Show column wise %ge of NaN values they contain
st.subheader('Data')
st.write(df.head())
st.write(df.describe())
st.write('Column wise %ge of NaN values they contain:')
for i in df.columns:
    st.write(i, ' - ', df[i].isna().mean()*100)

# Choose correlated columns
cormap = df.corr()
st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cormap, annot=True)
st.pyplot(fig)

def get_correlated_col(cor_data, threshold):
    """Function to get columns that are correlated with the target"""
    feature=[]
    value=[]
    for i, index in enumerate(cor_data.index):
        if abs(cor_data[index]) > threshold:
            feature.append(index)
            value.append(cor_data[index])
    df = pd.DataFrame(data=value, index=feature, columns=['corr value'])
    return df.index

top_correlated_cols = get_correlated_col(cormap['Close'], 0.60)
df = df[top_correlated_cols]
X = df.drop(['Close'], axis=1)
y = df['Close']

# Scale the data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

# Train the Linear Regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
acc_linreg = r2_score(y_test, y_pred_linreg)

# Train the CNN model
X_train_cnn = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
cnn_model = Sequential()
cnn_model.add(Conv1D(32, kernel_size=(3,), padding='same', activation='relu', input_shape=(X_train_cnn.shape[1],1)))
cnn_model.add(Conv1D(64, kernel_size=(3,), padding='same', activation='relu'))
cnn_model.add(Conv1D(128, kernel_size=(5,), padding='same', activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(50,activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_cnn, y_train, epochs=50, verbose=0)

st.subheader('Linear Regression Model')
st.write(f'Accuracy: {acc_linreg}')
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='True')
plt.plot(y_pred_linreg, label='Linear Regression')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()

st.subheader('CNN Model')
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = y_pred_cnn.reshape(-1)
acc_cnn = r2_score(y_test, y_pred_cnn)
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='True')
plt.plot(y_pred_cnn, label='CNN')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()
st.write(f'Accuracy: {acc_cnn}')
'''
'''
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
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

#Describing here
st.subheader('Data from 2010-2019')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting  the Data into Testing and Training

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)






#Load my model
model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
 
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

