import os
import json
import openai
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')
openai.api_key = os.getenv('OPEN_AI_API_KEY')

# Stock-related functions
def get_stock_price(ticker):
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

def calculate_RSI(ticker):
    """Calculate the Relative Strength Index (RSI) for a given ticker."""
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    emaUp = up.ewm(com=14-1, adjust=False).mean()
    emaDown = down.ewm(com=14-1, adjust=False).mean()
    rs = emaUp / emaDown
    return str(100 - (100 / (1 + rs)).iloc[-1])

def calculate_MACD(ticker):
    """Calculate the MACD for a given stock ticker."""
    data = yf.Ticker(ticker).history(period='1y').Close
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    MACD = short_ema - long_ema
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'

def plot_stock_price(ticker):
    """Plot the stock price for the last year given the ticker symbol of a company."""
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over the Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

# Function descriptions
functions = [
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    {
        'name': 'plot_stock_price',
        'description': 'Plot the stock price for the last year given the ticker symbol of a company',
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
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Streamlit app title
st.title('Stocks Assistant')

# User input
user_input = st.text_input('Your input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': f'{user_input}'})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response['choices'][0]['message']
        if response_message.get('function_call'):
            function_name = response_message['function_call']['name']
            function_args = json.loads(response_message['function_call']['arguments'])
            if function_name in ['get_stock_price', 'calculate_RSI', 'calculate_MACD', 'plot_stock_price']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculate_SMA', 'calculate_EMA']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plot_stock_price':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append(
                    {
                        'role': 'function',
                        'name': function_name,
                        'content': function_response
                    }
                )
                second_response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0613',
                    messages=st.session_state['messages']
                )
                st.text(second_response['choices'][0]['message']['content'])
                st.session_state['messages'].append({'role': 'assistant', 'content': second_response['choices'][0]['message']['content']})
        else:
            st.error("I don't know, I can only answer questions about the stock market")
            st.session_state['messages'].append({'role': 'assistant', 'content': "I'm not sure how to answer that."})

    except Exception as e:
        st.text('Please try again.')
