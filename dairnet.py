from sklearn.covariance import EllipticEnvelope
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import pickle as pk

dir = "/weka/Projects/dairnet/code/united/examples/ChoulesAnon/data/anon/"
dr = pd.read_csv(f"{dir}/train_10min.csv")
X = dr.iloc[:, 3:27]
X = X.drop_duplicates(subset = list(X)[1:], keep=False)
X.describe().transpose()
X = X.dropna()
X = X.to_numpy()

outliers_fraction = 0.01
algorithm = EllipticEnvelope(contamination=outliers_fraction, random_state=42)
algorithm.fit(X)

y_pred = algorithm.predict(X)
np.mean(y_pred)
["#377eb8", "#ff7f00"]


dt = pd.read_csv(f"{dir}/test_system_anomaly_10min.csv")
Y = dt.iloc[:, 3:27]
Y = Y.drop_duplicates(subset = list(Y)[1:], keep=False)
Y.describe().transpose()
Y = Y.dropna()
Y = Y.to_numpy()
y_pred = algorithm.predict(Y)
np.mean(y_pred)

dy = pk.load(f"{dir}/test_system_anomaly_10min_indices.pkl")

