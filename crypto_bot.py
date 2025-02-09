import requests
import time
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
# import pandas_ta as ta
from coinbase.wallet.client import Client
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.express as px
import backtrader as bt
from textblob import TextBlob
import numpy as np
np.nan



# 🔹 Load Environment Variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# 🔹 Connect to Coinbase
client = Client(COINBASE_API_KEY, COINBASE_API_SECRET)

def get_price(currency_pair="BTC-USD"):
    price = client.get_spot_price(currency_pair=currency_pair)
    return price['amount']

print(get_price())

# 🔹 Set Up Logging
logging.basicConfig(filename="trade_log.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# 🔹 AI Prediction Model with RandomForest

def ai_price_prediction(df):
    df['SMA'] = df['Close'].rolling(window=7).mean()
    df['EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Future_Trend'] = np.where(df['Price_Change'].shift(-1) > 0, 'Bullish', 'Bearish')
    features = df[['SMA', 'EMA', 'Price_Change']].dropna()
    labels = df['Future_Trend'].dropna()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features[:-1], labels[:-1])
    return model.predict(features[-1:].values.reshape(1, -1))[0]

# 🔹 Market Sentiment Analysis using TextBlob

def get_sentiment():
    tweets = ["Bitcoin is rising!", "Crypto crash incoming!", "Ethereum looking strong!"]
    sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
    return "Bullish" if sentiment_score > 0 else "Bearish"

sentiment = get_sentiment()
logging.info(f"Market Sentiment: {sentiment}")

# 🔹 Streamlit Dashboard

def show_dashboard():
    st.title("Crypto Portfolio Tracker 📊")
    prices = {"BTC": float(get_price("BTC-USD")), "ETH": float(get_price("ETH-USD")), "SOL": float(get_price("SOL-USD"))}
    df = pd.DataFrame(prices.items(), columns=["Asset", "Price"])
    df["PnL (%)"] = ((df["Price"] - df["Price"].shift(1)) / df["Price"].shift(1)) * 100
    
    fig = px.bar(df, x="Asset", y="Price", title="Current Prices")
    st.plotly_chart(fig)
    st.table(df)

if __name__ == "__main__":
    show_dashboard()
