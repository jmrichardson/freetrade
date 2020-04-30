import pandas as pd


class Kibot():

    def __init__(self, path, nrows=None):
        self.df = pd.read_csv(path, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'], nrows=nrows)
        self.df['date'] = self.df.date + ' ' + self.df.time
        self.df.drop(columns=['time'], inplace=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

