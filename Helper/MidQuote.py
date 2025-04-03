# class to calculate mid quote price
## test_duh7246
class MidQuote:
    def __init__(self,data):
        self.mid_quotes = []
        self.timestamps = []
        for i in range(data.getN()):
            bid_price = data.getBidPrice(i)
            ask_price = data.getAskPrice(i)
            timestamp = data.getMillisFromMidn(i)

            if bid_price is not None and ask_price is not None:
                mid_quote = (bid_price + ask_price) / 2
                self.mid_quotes.append(mid_quote)
                self.timestamps.append(timestamp)
    def getN(self):
        return len(self.mid_quotes)

    def getTimestamp(self,i):
        return self.timestamps[i]

    def getPrice(self,i):
        return self.mid_quotes[i]