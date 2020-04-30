import freetrade as ft
from sklearn.pipeline import Pipeline

# Format (date, open, high, low, close, volume)
ibm = ft.Kibot('data/kibot/IBM.txt', nrows=10000)

pipeline = Pipeline([
    ('remove_outliers', ft.OutlierStdRemove(10)),
    ('triple_barrier_labeling', ft.TripleBarierLabeling(close_name='close')),
])
pipe_out = pipeline.fit_transform(ibm.df)

