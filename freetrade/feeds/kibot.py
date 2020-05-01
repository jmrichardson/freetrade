import pandas as pd
from freetrade.feeds.feed import Feed


class Kibot(Feed):

    def __init__(self, path, nrows=None):
        # X must have format of open, hich, low, close volume - with datetime index
        self.X = pd.read_csv(path, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'], nrows=nrows)
        self.X['date'] = self.X.date + ' ' + self.X.time
        self.X.drop(columns=['time'], inplace=True)
        self.X['date'] = pd.to_datetime(self.X['date'])
        self.X.set_index('date', inplace=True)

