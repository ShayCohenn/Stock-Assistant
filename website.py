import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date
from prophet.plot import plot_plotly
from functions import chatbot, get_company_name, load_data, time_pick, plot_raw_data, predict_stock, plot_prediction

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

selected_stock = st.text_input("Select a stock", value="AAPL")

timeframe = st.selectbox("Select timeframe", ("1 day", "1 week", "1 month", "3 months", "6 months", "1 year", "2 years", "5 years"))

period = time_pick(timeframe)

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)

if data is not None:
    data_load_state.text("Data loaded!")
    company_name = get_company_name(selected_stock)

    st.subheader(f"{company_name}({selected_stock.upper()}) Data")
    st.write(data.iloc[::-1])

    st.plotly_chart(plot_raw_data(data))

    forecast, forecast_display, model = predict_stock(data, period)

    st.subheader("Forecast data")
    st.write(forecast_display.iloc[::-1])

    st.subheader("prediction vs actual price")

    st.plotly_chart(plot_prediction(data, forecast))

    st.subheader("Forecast graph")
    st.write("with margin of error visualized")
    st.plotly_chart(plot_plotly(model, forecast))
else:
    st.error(f"{selected_stock} not found!")
    data_load_state.empty()