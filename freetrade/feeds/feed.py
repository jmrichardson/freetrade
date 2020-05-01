import pandas as pd
import numpy as np
import mlfinlab as ml
from sklearn import preprocessing


class Feed:

    def __init__(self):
        self.X = pd.DataFrame()
        self.y = pd.Series()

    def outlierStdRemove(self, std_threshold):
        self.X = self.X[self.X.apply(lambda x: np.abs(x - x.mean()) / x.std() < std_threshold).all(axis=1)]

    def dollarBars(self, frac=.02, batch_size=100000, verbose=False):
        # Set threshold to fraction of average daily dollar value (default 1/50)
        df = self.X[['close','volume']].resample('D').mean().dropna()
        df['dollar'] = (df.close * df.volume)
        threshold = df.dollar.mean()*frac

        # Sample by dollar value
        df = self.X.reset_index()
        df = df.rename(columns={'date': 'date_time'})
        df = df[['date_time', 'close', 'volume']]
        self.X = ml.data_structures.standard_data_structures.get_dollar_bars(df, threshold=threshold,
            batch_size=batch_size, verbose=verbose)
        self.X.set_index('date_time', inplace=True)

    def tripleBarierLabeling(self, volatility_lookback=50, volatility_scaler=1, triplebar_num_days=3,
        triplebar_pt_sl=[1, 1], triplebar_min_ret=0.003, num_threads=1):

        # extract close series
        close = self.X['close']

        # Compute volatility
        daily_vol = ml.util.get_daily_vol(close, lookback=volatility_lookback)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(close, threshold=daily_vol.mean() * volatility_scaler)

        # Compute vertical barrier
        vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=close,
            num_days=triplebar_num_days)

        # tripple barier events
        triple_barrier_events = ml.labeling.get_events( close=close, t_events=cusum_events, pt_sl=triplebar_pt_sl,
            target=daily_vol, min_ret=triplebar_min_ret, num_threads=num_threads,
            vertical_barrier_times=vertical_barriers)

        # labels
        labels = ml.labeling.get_bins(triple_barrier_events, close)
        labels = ml.labeling.drop_labels(labels)

        # merge labels and triple barrier events
        triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        triple_barrier_info.dropna(inplace=True)

        self.X = self.X.reindex(triple_barrier_info.index)
        self.y = triple_barrier_info.bin

