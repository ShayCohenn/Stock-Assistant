# Stock Assistant
### Stock Assistant website.<br>
### Includes a chatbot powered by OpenAI, a detailed view of a stock, graphs and a prediction for a period of time
<hr>

### <a href="https://stock-assistant-1.streamlit.app/">Live Website</a> hosted on streamlit
## Packages used:
• <a href="https://pypi.org/project/yfinance/">yfinance</a> by Yahoo, for the stocks API <br>
• <a href="https://pypi.org/project/streamlit/">Streamlit</a> for the website and hosting<br>
• <a href="https://openai.com/">OpenAI</a> for the chatbot API, I used the gpt-3.5-turbo-1106 model<br>
• <a href="https://github.com/facebook/prophet">Meta's prophet</a> for predicting the stock price<br>
• <a href="https://plotly.com/">plotly</a> for creating the graphs.
## Features
### • Chatbot:
• Getting the value of any stock<br>
• Getting the ticker symbol providing a company's name<br>
• Getting indicators of any stock, for example:<br>
• Simple Moving Average (SMA),<br>
• Exponential Moving Average (EMA),<br>
• Relative Strength Index (RSI) and MACD <br>
### Detailed view:
• Graph for any period of time after 2010 of the selected stock<br>
• Table and a downloadable CSV file for any period of time after 2012 of the selected stock<br>
• Price prediction for a selected time(1d, 1 week, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years)<br>

## Installation:
### If you want to edit the code and create your own chatbot this is how:
### clone the repository:
```
git clone https://github.com/ShayCohenn/Stock-Assistant
```
### cd into the folder
```
cd Stock-Assistant
```
### create your .env.local file with your OpenAI api key like this 
```bash
OPEN_AI_API_KEY = ""
```
### create virtual enviroment
```
python -m virtualenv env
```
### activate the virtualenv
```
.\env\Scripts\activate
```
### install the dependencies
```
pip install -r requirements.txt
```
### run the program
```
streamlit run .\website.py
```