import freetrade as ft
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Load data
ibm = ft.Kibot('data/kibot/IBM.txt', nrows=100000)

# Preprocess
ibm.outlierStdRemove(10)
ibm.dollarBars()
ibm.tripleBarierLabeling()
ibm.trainTestSplit()

# Apply pipelines
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('logisticRegression', LogisticRegression())
])
fit = pipeline.fit(ibm.X_train, ibm.y_train)


