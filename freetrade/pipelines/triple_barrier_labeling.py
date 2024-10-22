import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import mlfinlab as ml


class TripleBarierLabeling(BaseEstimator, TransformerMixin):

    def __init__(self, close_name='close', volatility_lookback=50,
                 volatility_scaler=1, triplebar_num_days=3,
                 triplebar_pt_sl=[1, 1], triplebar_min_ret=0.003,
                 num_threads=1):
        # hyperparameters for all functions
        self.close_name = close_name
        self.volatility_lookback = volatility_lookback
        self.volatility_scaler = volatility_scaler
        self.triplebar_num_days = triplebar_num_days
        self.triplebar_pt_sl = triplebar_pt_sl
        self.triplebar_min_ret = triplebar_min_ret
        self.num_threads = num_threads

    def fit(self, X, y=None):
        # extract close series
        close = X.loc[:, self.close_name]
        close = X['close']

        # Compute volatility
        daily_vol = ml.util.get_daily_vol(close, lookback=self.volatility_lookback)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(
            close,
            threshold=daily_vol.mean() * self.volatility_scaler)

        # Compute vertical barrier
        vertical_barriers = ml.labeling.add_vertical_barrier(
            t_events=cusum_events,
            close=close,
            num_days=self.triplebar_num_days)

        # tripple barier events
        triple_barrier_events = ml.labeling.get_events(
            close=close,
            t_events=cusum_events,
            pt_sl=self.triplebar_pt_sl,
            target=daily_vol,
            min_ret=self.triplebar_min_ret,
            num_threads=self.num_threads,
            vertical_barrier_times=vertical_barriers)

        # labels
        labels = ml.labeling.get_bins(triple_barrier_events, close)
        labels = ml.labeling.drop_labels(labels)

        # merge labels and triple barrier events
        self.triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        self.triple_barrier_info.dropna(inplace=True)

        return self

    def transform(self, X):
        # subsample
        X = X.reindex(self.triple_barrier_info.index)
        y = self.triple_barrier_info.bin

        return [X, y]
