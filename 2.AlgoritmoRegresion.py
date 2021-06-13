from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

boston = load_boston()

boston.keys()

boston.data
boston['data']

boston.target
boston.data.shape

boston.target.shape

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)

X_train.shape

X_test.shape

y_test.shape

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)

del knn

rl = LinearRegression()
rl.fit(X_train, y_train)

rl.score(X_test, y_test)

del rl

ridge = Ridge()
ridge.fit(X_train, y_train)
ridge.score(X_test, y_test)

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge.score(X_test, y_test)