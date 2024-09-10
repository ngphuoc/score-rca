from sklearn.covariance import EllipticEnvelope
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import pickle as pk
from sklearn import metrics

def indexin(y, x):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    # indices = index[sorted_index]
    # return indices
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] == y
    return yindex, mask

dir = "/weka/Projects/dairnet/code/united/examples/ChoulesAnon/data/anon"
dr = pd.read_csv(f"{dir}_bak/train_10min.csv")
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

dt = pd.read_csv(f"{dir}/test_system_anomaly_10min_0.2.csv")
y_timeidx = pk.load(open(f"{dir}/test_system_anomaly_10min_0.2_indices.pkl",'rb'))
y_idx, mask = indexin(y_timeidx, dt["time_idx"].to_numpy())
np.mean(mask)
Y = dt.iloc[:, 3:27]
Y = Y.drop_duplicates(subset = list(Y)[1:], keep=False)
Y.describe().transpose()
Y = Y.dropna()
Y = Y.to_numpy()
y_pred = algorithm.predict(Y)
np.mean(y_pred)
len(y_pred)
y_gt = np.zeros(len(y_pred))
y_gt[y_idx] = 1
np.mean(y_gt)
fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_pred, pos_label=1)
metrics.auc(fpr, tpr)
np.mean(y_gt == y_pred)

Y[y_idx]
b, d = Y.shape
np.setdiff
_idx = np.setdiff1d(np.arange(b), y_idx)
Y

dy = pk.load(f"{dir}/test_system_anomaly_10min_indices.pkl")

# x = np.array([3, 5, 7, 1, 9, 8, 6, 6])
# y = np.array([2, 1, 5, 10, 100, 6])
# idx, mask = indexin(x, y)
# idx[mask]

