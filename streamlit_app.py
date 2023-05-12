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

st.set_page_config(page_title="Stock Predictor", page_icon=":chart_with_upwards_trend:")

# Decision Tree Streamlit app
def decision_tree_app():
    st.title("Decision Tree Regression")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start = '2010-01-01'
    end = '2019-12-31'
    df = yf.download(ticker, start, end)
    # Rest of the code for data summary, visualization, and prediction

# SVM Streamlit app
def svm_app():
    st.title("Support Vector Machine (SVM)")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-05-03"))
    # Rest of the code for data fetching, preprocessing, training, and prediction

# LSTM Streamlit app
def lstm_app():
    st.title("Long Short-Term Memory (LSTM)")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start = '2010-01-01'
    end = '2019-12-31'
    df = yf.download(ticker, start, end)
    # Rest of the code for data description, visualization, preprocessing, model loading, and prediction

# Linear Regression & CNN Streamlit app
def linear_cnn_app():
    st.title("Linear Regression and CNN")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL for Apple)", "AAPL")
    start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2009-01-01"))
    end_date = st.sidebar.date_input("End date:", value=pd.to_datetime("2022-12-31"))
    # Rest of the code for data fetching, correlation analysis, feature selection, scaling, train-test split,
    # linear regression training and prediction, CNN model training and prediction, and result visualization

# App selection
app_options = {
    "Decision Tree Regression": decision_tree_app,
    "SVM": svm_app,
    "LSTM": lstm_app,
    "Linear Regression & CNN": linear_cnn_app
}

app_selection = st.sidebar.selectbox("Select the Stock Predictor App", list(app_options.keys()))

if app_selection:
    app_options[app_selection]()  # Execute the selected app
