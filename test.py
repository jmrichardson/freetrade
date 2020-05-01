import freetrade as ft
from sklearn.pipeline import Pipeline

# Load data
ibm = ft.Kibot('data/kibot/IBM.txt', nrows=100000)

# Preprocess
ibm.outlierStdRemove(10)
ibm.dollarBars()
ibm.tripleBarierLabeling()

# Apply pipelines
# pipeline = Pipeline([
# ])
# X = pipeline.fit_transform(ibm.df)

