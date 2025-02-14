# System & Utility Imports
import os
import sys
import time
import threading
from datetime import datetime
import logging

# External Libraries
import requests
from dotenv import load_dotenv
from coinbase.wallet.client import Client
from flask import Flask, render_template, jsonify, request, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("API Key and Secret are required. Check your .env file.")

client = Client(API_KEY, API_SECRET)

# Parameters for buying the dip
LOOKBACK_PERIOD = 5  # Check last 5 minutes
PRICE_DROP_PERCENTAGE = 2  # Minimum % drop to consider as a dip
BUY_AMOUNT_USD = 2  # Amount to buy in USD
CHECK_INTERVAL = 60  # Check every 60 seconds

app = Flask(__name__)

# Security: Rate limiting to prevent abuse
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour", "10 per minute"])

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

price_history = {"BTC": [], "ETH": [], "SOL": []}
time_history = {"BTC": [], "ETH": [], "SOL": []}  # Track timestamps
latest_prices = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
status_message = "Monitoring market..."
dip_detected = {"BTC": False, "ETH": False, "SOL": False}

def get_crypto_price(symbol):
    try:
        response = requests.get(f'https://api.coinbase.com/v2/prices/{symbol}-USD/spot', timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['data']['amount'])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {symbol} price: {e}")
        return latest_prices.get(symbol, 0.0)

def buy_crypto(symbol, amount):
    global status_message, dip_detected
    try:
        account = client.get_primary_account()
        payment_method = client.get_payment_methods()[0]  # Use first available method
        txn = account.buy(amount=str(amount), currency='USD', payment_method=payment_method['id'])
        status_message = f"Bought ${amount} worth of {symbol} at {txn['total']} USD/{symbol}"
        dip_detected[symbol] = False
        logging.info(status_message)
    except Exception as e:
        logging.error(f"Failed to execute buy order: {e}")
        status_message = "Error executing purchase. Check logs."

def monitor_market():
    global latest_prices, status_message, dip_detected
    while True:
        for symbol in ["BTC", "ETH", "SOL"]:
            latest_prices[symbol] = get_crypto_price(symbol)
            current_time = datetime.now().strftime("%H:%M:%S")
            price_history[symbol].append(latest_prices[symbol])
            time_history[symbol].append(current_time)
            
            if len(price_history[symbol]) > LOOKBACK_PERIOD:
                price_history[symbol].pop(0)
                time_history[symbol].pop(0)

            if len(price_history[symbol]) == LOOKBACK_PERIOD:
                max_price = max(price_history[symbol])
                price_drop = ((max_price - latest_prices[symbol]) / max_price) * 100
                
                if price_drop >= PRICE_DROP_PERCENTAGE:
                    status_message = f"Dip detected for {symbol}: {price_drop:.2f}% drop. Click 'Buy' to confirm."
                    dip_detected[symbol] = True
                else:
                    status_message = f"No significant dip detected for {symbol}. Current drop: {price_drop:.2f}%"
                    dip_detected[symbol] = False
        
        time.sleep(CHECK_INTERVAL)

@app.route('/')
@limiter.limit("5 per second")
def home():
    return render_template('index.html')

@app.route('/data')
@limiter.limit("10 per second")
def data():
    return jsonify({
        "latest_prices": latest_prices,
        "status": status_message,
        "price_history": price_history,
        "time_history": time_history,
        "dip_detected": dip_detected
    })

@app.route('/buy/<symbol>', methods=['POST'])
@limiter.limit("3 per minute")
def buy(symbol):
    global dip_detected
    if symbol not in ["BTC", "ETH", "SOL"]:
        abort(400, "Invalid symbol")
    
    if dip_detected.get(symbol, False):
        buy_crypto(symbol, BUY_AMOUNT_USD)
        return jsonify({"message": f"Purchase confirmed for {symbol}!", "status": status_message})
    return jsonify({"message": f"No dip detected for {symbol} purchase.", "status": status_message})

if __name__ == "__main__":
    threading.Thread(target=monitor_market, daemon=True).start()
    try:
        app.run(debug=False, host='0.0.0.0', port=5001)  # Changed port to avoid conflicts
    except OSError as e:
        logging.error(f"Port 5001 is in use. Please choose a different port or close the conflicting process.")
