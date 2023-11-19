import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly
from functions import chatbot, time_pick, plot_raw_data, predict_stock, plot_prediction

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title("Stock Assistant")

st.header("Chatbot")

user_input = st.text_input('Your input:')

if user_input:
    response = chatbot([{'role': 'user', 'content': f'{user_input}'}])
    st.write(response)

st.divider()

st.header("Stock details and predictions")

stocks = ("AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "^GSPC", "BAC")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

timeframe = st.selectbox("Select timeframe", ("1 day", "1 week", "1 month", "3 months", "6 months", "1 year", "2 years"))

period = time_pick(timeframe)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data loaded!")

st.subheader("Raw Data")
st.write(data.iloc[::-1])

st.plotly_chart(plot_raw_data(data))

forecast, model = predict_stock(data, period)

st.subheader("Forecast data")
st.write(forecast.iloc[::-1])

st.subheader("prediction vs actual price")

st.plotly_chart(plot_prediction(data, forecast))

st.subheader("Forecast graph")
st.write("with error margin")
st.plotly_chart(plot_plotly(model, forecast))