import requests
import time
import os
import logging
import pandas as pd
import numpy as np
from coinbase.wallet.client import Client
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.express as px

# Load Environment Variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

# Configure Logging
logging.basicConfig(filename="trade_log.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Connect to Coinbase API
def connect_coinbase():
    try:
        return Client(COINBASE_API_KEY, COINBASE_API_SECRET)
    except Exception as e:
        logging.error(f"Error connecting to Coinbase API: {e}")
        return None

client = connect_coinbase()

# Fetch Price from Coinbase API with Caching and Retry Mechanism
def get_price(currency_pair="BTC-USD", retries=3, delay=5):
    if not client:
        return None
    for attempt in range(retries):
        try:
            price = client.get_spot_price(currency_pair=currency_pair)
            return float(price['amount'])
        except Exception as e:
            logging.error(f"Error fetching price for {currency_pair}: {e}")
            time.sleep(delay)
    return None

# Compute Technical Indicators
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_macd(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def compute_indicators(df):
    df['SMA'] = df['Close'].rolling(window=7).mean()
    df['EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df = compute_rsi(df)
    df = compute_macd(df)
    df.dropna(inplace=True)
    return df

# AI-based Price Prediction
def ai_price_prediction(df):
    df = compute_indicators(df)
    df['Future_Trend'] = np.where(df['Close'].pct_change().shift(-1) > 0, 'Bullish', 'Bearish')
    features = df[['SMA', 'EMA', 'RSI', 'MACD', 'Signal_Line']]
    labels = df['Future_Trend']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"AI Model Accuracy: {accuracy:.2f}")
    
    return model.predict(features[-1:].values.reshape(1, -1))[0]

# Decision Making for Trading
def trade_decision():
    btc_price = get_price("BTC-USD")
    if btc_price is None:
        return "Error fetching BTC price."
    
    df = pd.DataFrame({'Close': [btc_price]})
    df = compute_indicators(df)
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    
    if btc_price < 40000 and rsi < 30 and macd > signal:
        decision = "Strong Buy"
    elif btc_price < 40000:
        decision = "Buy"
    elif btc_price > 60000 and rsi > 70 and macd < signal:
        decision = "Strong Sell"
    elif btc_price > 60000:
        decision = "Sell"
    else:
        decision = "Hold"
    
    logging.info(f"Trade decision: {decision} at BTC price: {btc_price}")
    return decision

# Streamlit Dashboard
def show_dashboard():
    st.title("📊 Crypto Portfolio & Trading Bot")
    
    assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    try:
        prices = {asset.split('-')[0]: get_price(asset) or 0 for asset in assets}
        df = pd.DataFrame(prices.items(), columns=["Asset", "Price"])
        df["PnL (%)"] = df["Price"].pct_change() * 100
        
        fig_price = px.bar(df, x="Asset", y="Price", title="Current Prices", text="Price")
        st.plotly_chart(fig_price)
        
        decision = trade_decision()
        st.subheader(f"Trade Decision: {decision}")
        
        btc_df = pd.DataFrame({'Close': [get_price("BTC-USD")]})
        btc_df = compute_indicators(btc_df)
        
        fig_rsi = px.line(btc_df, y="RSI", title="RSI Trend (BTC)")
        fig_macd = px.line(btc_df, y=["MACD", "Signal_Line"], title="MACD & Signal Line (BTC)")
        
        st.plotly_chart(fig_rsi)
        st.plotly_chart(fig_macd)
        
        st.table(df)
    
    except Exception as e:
        logging.error(f"Error in dashboard: {e}")
        st.error("Failed to load dashboard. Please try again later.")

if __name__ == "__main__":
    show_dashboard()