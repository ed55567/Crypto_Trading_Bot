from dotenv import load_dotenv
import os

load_dotenv()
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_KEY = os.getenv("COINBASE_API_SECRET")

# Connect to Coinbase & Fetch Real-Time Data

from coinbase.wallet.client import Client

client = Client(COINBASE_API_KEY, COINBASE_API_SECRET)

def get_price(currency_pair="BTC-USD"):
    rice = client.get_spot_price(currency_pair=currency_pair)
    return price['amount']

print(get_price())

