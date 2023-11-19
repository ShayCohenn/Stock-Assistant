import os
import json
import pandas as pd
import yfinance as yf
from openai import OpenAI
from prophet import Prophet
from dotenv import load_dotenv
from plotly import graph_objs as go

# Stock-related functions
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
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD
}

def time_pick(time):
    if time == "1 day":
        return 1
    elif time == "1 week":
        return 7
    elif time == "1 month":
        return 30
    elif time == "3 months":
        return 90
    elif time == "6 months":
        return 180
    elif time == "1 year":
        return 365
    elif time == "2 years":
        return 730
    else:
        return 1

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close", line=dict(color='blue')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    return fig

def predict_stock(data, period):
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['ds'] = forecast['ds'].dt.date

    return forecast, model

def plot_prediction(data, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=data['Close'], name='actual stock', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='red')))
    fig.update_layout(title_text="Forecast Data", xaxis_rangeslider_visible=True)
    return fig

load_dotenv('.env.local')

client = OpenAI(api_key=os.getenv('OPEN_AI_API_KEY'))

def chatbot(messages):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-1106",
        messages = messages,
        tools = tools,
        tool_choice = "auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        messages.append(response_message)
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
        second_response = client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            messages = messages,
        )
        return second_response.choices[0].message.content
    else:
        return "I'm not sure how to answer that."