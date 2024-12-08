using Flux, JLD2, DataFrames, CSV
include("lib/utils.jl")
include("imports.jl")

dir = "/weka/Projects/dairnet/code/united/examples/ChoulesAnon/data/anon/"
dr = CSV.read("$dir/train_10min.csv", DataFrame)
cols = names(dr)
jj = findall(contains("Sensor22"), cols)
d22 = dr[!, jj]
@> eachcol(d22) unique.() length.()

dr[!, 2]
dr[!, :group_id]
dr[!, :time_idx]
describe(dr)
@> dr[!, :group_id] unique

@unpack EllipticEnvelope = pyimport("sklearn.covariance")

outliers_fraction = 0.01
algorithm = EllipticEnvelope(contamination=outliers_fraction, random_state=42)
X = Array(d22)

import numpy as np
from sklearn.impute import KNNImputer
nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X)

algorithm.fit(X)
t1 = time.time()
plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
y_pred = algorithm.fit(X).predict(X)
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

