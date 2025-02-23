import ccxt

# Initialize exchange objects
binance = ccxt.binance()
coinbase = ccxt.coinbase()
kraken = ccxt.kraken()

# Fetch current prices
def get_price(symbol="BTC/USDT"):
    binance_price = binance.fetch_ticker(symbol)['last']
    coinbase_price = coinbase.fetch_ticker(symbol)['last']
    kraken_price = kraken.fetch_ticker(symbol)['last']

    return {
        "Binance": binance_price,
        "Coinbase": coinbase_price,
        "Kraken": kraken_price
    }

print(get_price())