import os
import json
import pandas as pd
import yfinance as yf
import streamlit as st
from openai import OpenAI
from datetime import date
from prophet import Prophet
from dotenv import load_dotenv
from plotly import graph_objs as go

# ----------------------------- Chatbot functions --------------------------------
def get_stock_price(ticker, window):
    """Get the latest stock price given the ticker symbol of a company."""
    data = yf.Ticker(ticker).history(period='1y')
    return str(data.iloc[-1].Close)

def calculate_SMA(ticker, window):
    """Calculate the Simple Moving Average (SMA) for a given ticker and a window."""
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    """Calculate the Exponential Moving Average (EMA) for a given ticker and a window."""
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker, window):
    """Calculate the Relative Strength Index (RSI) for a given ticker."""
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    emaUp = up.ewm(com=14-1, adjust=False).mean()
    emaDown = down.ewm(com=14-1, adjust=False).mean()
    rs = emaUp / emaDown
    return str(100 - (100 / (1 + rs)).iloc[-1])

def calculate_MACD(ticker, window):
    """Calculate the MACD for a given stock ticker."""
    data = yf.Ticker(ticker).history(period='1y').Close
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    MACD = short_ema - long_ema
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD.iloc[-1]}, {signal.iloc[-1]}, {MACD_histogram.iloc[-1]}'

def get_company_symbol(ticker, window):
    """Get the company symbol"""
    company = yf.Ticker(ticker)
    symbol = company.info['symbol']
    return symbol

# ---------------------------------- Formatting fucntions for OpenAI API ----------------------------------

tools = [
    {
        'type': 'function',
        'function':{
            'name': 'get_stock_price',
            'description': 'Gets the latest stock price given the ticker symbol of a company.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    }
                },
                'required': ['ticker']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_SMA',
            'description': 'Calculates the Simple Moving Average (SMA) for a given ticker and a window.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    },
                    'window': {
                        'type': 'integer',
                        'description': 'The timeframe to consider when calculating the SMA.'
                    }
                },
                'required': ['ticker', 'window'],
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_EMA',
            'description': 'Calculates the Exponential Moving Average (EMA) for a given ticker and a window.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    },
                    'window': {
                        'type': 'integer',
                        'description': 'The timeframe to consider when calculating the EMA.'
                    }
                },
                'required': ['ticker', 'window'],
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_RSI',
            'description': 'Calculates the Relative Strength Index (RSI) for a given ticker.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    }
                },
                'required': ['ticker'],
            },
        }
    },
    {        
        'type': 'function',
        'function': {
            'name': 'calculate_MACD',
            'description': 'Calculate the MACD for a given stock ticker',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    },
                },
                'required': ['ticker'],
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_company_symbol',
            'description': 'Get the company symbol',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type':'string',
                        'description': 'The stock ticker symbol for a company (e.g., AAPL for Apple).'
                    },
                },
               'required': ['ticker'],
            },
        }
    }
]

# ----------------------------- Connecting the formatted functions to the actual functions ----------------------------------

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'get_company_symbol': get_company_symbol
}

# ------------------------------ Chatbot ----------------------------------

# Loading the .env.local file
load_dotenv('.env.local')

# Initializing the client
client = OpenAI(api_key=os.getenv('OPEN_AI_API_KEY'))

def chatbot(messages):
    """
    We give the AI the message from the user and the functions 
    to get the AI's choice if function to execute
    """
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-1106",
        messages = messages,
        tools = tools,
        tool_choice = "auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # If the AI found a fitting function → proceed
    if tool_calls:
        messages.append(response_message)
        # Send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                ticker = function_args.get("ticker"),
                window = function_args.get("window")
            )
            messages.append(
                {
                    "tool_call_id":tool_call.id,
                    "role":"tool",
                    "name":function_name,
                    "content":function_response
                }
            )
            # Get a new response from the model where it can see the function response
        second_response = client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            messages = messages,
        )
        return second_response.choices[0].message.content
    # If the AI didn't find a fitting function → return an error message
    else:
        return "I'm not sure how to answer that."

# ---------------------------------- Main ----------------------------------

# Coverting the time selected to amount of days
def time_pick(time):
    amount, period = time.split(" ")
    amount = int(amount)
    if period == "day" or period == "days":
        return amount
    elif period == "week" or period == "weeks":
        return amount * 7
    elif period == "month" or period == "months":
        return amount * 30
    elif period == "year" or period == "years":
        return amount * 365

# Loading the data and caching it
@st.cache_data
def load_data(ticker):
    try:
        START = "2012-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")
        company = yf.Ticker(ticker).info
        if company is not None:
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data['Date'] = data['Date'].dt.date
            return data
        return None
    except:
        return None

def get_company_name(ticker):
    company = yf.Ticker(ticker)
    name = company.info['longName']
    return name

# Graph the stock's data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close", line=dict(color='blue')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    return fig

# Prediction method
def predict_stock(data, period):
    def calculate_rsi(data, column='Close', length=15):
        close_price = data[column]
        price_diff = close_price.diff()
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        avg_gain = gain.rolling(window=length, min_periods=1).mean()
        avg_loss = loss.rolling(window=length, min_periods=1).mean()

        relative_strength = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + relative_strength))

        return rsi
    def calculate_ema(data, column='Close', length=20):
        return data[column].ewm(span=length, adjust=False).mean()
    
    data['RSI'] = calculate_rsi(data, column='Close', length=15)
    data['EMAF'] = calculate_ema(data, column='Close', length=20)
    data['EMAM'] = calculate_ema(data, column='Close', length=100)
    data['EMAS'] = calculate_ema(data, column='Close', length=150)

    df_train = data[["Date", "Close", "RSI", "EMAF", "EMAM", "EMAS"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['ds'] = forecast['ds'].dt.date

    # Removing the irrelevant columns
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Renaming the columns for display
    forecast_display = forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted Close Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})

    return forecast, forecast_display, model

# Graph the prediction
def plot_prediction(data, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=data['Close'], name='actual stock', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='red')))
    fig.update_layout(title_text="Forecast Data", xaxis_rangeslider_visible=True)
    return fig