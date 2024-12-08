from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#
# Load California housing data set
#
housing = fetch_california_housing()
X = housing.data
y = housing.target
#
# Create training/ test data split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#
# Instantiate MLPRegressor
#
nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(10, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=False
)
#
# Train the model
#
nn.fit(X_train, y_train)

import numpy as np
from sklearn.metrics import mean_squared_error

# Make prediction
pred = nn.predict(X_test)
#
# Calculate accuracy and error metrics
#
test_set_rsquared = nn.score(X_test, y_test)
test_set_rmse = np.sqrt(mean_squared_error(y_test, pred))
#
# Print R_squared and RMSE value
#
print('R_squared value: ', test_set_rsquared)
print('RMSE: ', test_set_rmse)

len(nn.coefs_)
nn.coefs_[0].shape
nn.coefs_[1].shape
nn.coefs_[2].shape

len(nn.intercepts_)
nn.intercepts_[0].shape
nn.intercepts_[1].shape
nn.intercepts_[2].shape

type(nn.intercepts_[0])

hidden = 100
n = 1000
z = np.random.normal(scale=0.1, size=n)
parents = [1, 2, 3]
pa_size = len(parents)
X = np.random.randn(n, pa_size)
W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
W1[np.random.rand(*W1.shape) < 0.5] *= -1
W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
W2[np.random.rand(hidden) < 0.5] *= -1
y = sigmoid(X @ W1) @ W2 + z
regressor = GaussianProcessRegressor()
regressor = MLPRegressor((100, ), "logistic")
regressor.fit(X,y)
assert regressor.coefs_[0].shape == W1.shape
np.ndindex(*regressor.coefs_[0].shape)
regressor.coefs_[0] = W1
regressor.coefs_[1] = W2.reshape(regressor.coefs_[1].shape)
regressor.intercepts_[0] = np.zeros_like(regressor.intercepts_[0])
regressor.intercepts_[1] = np.zeros_like(regressor.intercepts_[1])
regressor.partial_fit(X, y)

