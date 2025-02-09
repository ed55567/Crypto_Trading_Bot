import requests
import time
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
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




# 🔹 Load Environment Variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# 🔹 Connect to Coinbase with Exception Handling
try:
    client = Client(COINBASE_API_KEY, COINBASE_API_SECRET)
except Exception as e:
    logging.error(f"Error connecting to Coinbase API: {e}")
    client = None


def get_price(currency_pair="BTC-USD"):
    """Fetch the latest price from Coinbase API with error handling."""
    if not client:
        return None
    try:
        price = client.get_spot_price(currency_pair=currency_pair)
        return price['amount']
    except Exception as e:
        logging.error(f"Error fetching price for {currency_pair}: {e}")
        return None

# print(get_price())

# 🔹 Set Up Logging
logging.basicConfig(filename="trade_log.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# 🔹 AI Prediction Model with RandomForest

def ai_price_prediction(df):
    """AI Model for price trend prediction using Random Forest."""
    if 'Close' not in df.columns:
        logging.error("DataFrame missing 'Close' column for AI prediction.")
        return "Error"

    df['SMA'] = df['Close'].rolling(window=7).mean()
    df['EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)


    df['Future_Trend'] = np.where(df['Price_Change'].shift(-1) > 0, 'Bullish', 'Bearish')

    features = df[['SMA', 'EMA', 'Price_Change']]
    labels = df['Future_Trend']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features[:-1], labels[:-1])

    return model.predict(features[-1:].values.reshape(1, -1))[0]

# 🔹 Market Sentiment Analysis using TextBlob

def get_sentiment():
    """Analyze market sentiment based on tweets."""
    tweets = ["Bitcoin is rising!", "Crypto crash incoming!", "Ethereum looking strong!"]
    sentiment_score = np.mean([TextBlob(tweet).sentiment.polarity for tweet in tweets])
    return "Bullish" if sentiment_score > 0 else "Bearish"

# 🔹 Function to Send Telegram Alert
def send_telegram_message(message):
    """Send trade alerts via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram bot token or chat ID missing.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# 🔹 Streamlit Dashboard

def show_dashboard():
    """Streamlit dashboard to display cryptocurrency prices and AI predictions."""
    st.title("Crypto Portfolio Tracker 📊")

    assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
    prices = {asset.split('-')[0]: float(get_price(asset) or 0) for asset in assets}

    df = pd.DataFrame(prices.items(), columns=["Asset", "Price"])
    df["PnL (%)"] = df["Price"].pct_change() * 100  # Calculate % Change
    
    fig = px.bar(df, x="Asset", y="Price", title="Current Prices")
    st.plotly_chart(fig)
    st.table(df)

if __name__ == "__main__":
    show_dashboard()
