from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


#parametrs for using lasso and ridge
parameters = {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 500]}

dataset = fetch_california_housing()

dataframe = pd.DataFrame(dataset.data ,columns = dataset.feature_names)

dataframe['price'] = dataset.target

X = dataframe.drop('price',axis = 1)
Y = dataframe['price']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=2)

LinearModel = LinearRegression()
LinearModel.fit(X_train,Y_train)
Y_predicted = LinearModel.predict(X_test)

score = metrics.mean_absolute_error(Y_test,Y_predicted)

print(f"Evaluation matrics MAE for Linear regression is : {score}")

ridge = Ridge()
Ridge_model = GridSearchCV(ridge,parameters,scoring='neg_mean_absolute_error',cv = 5)
Ridge_model.fit(X_train,Y_train)
Y_predicted = Ridge_model.predict(X_test)

# print(type(Ridge_model.best_score_))

# score = metrics.mean_absolute_error(Y_test,Y_predicted)

print(f"Evaluation matrics MAE for Ridge regression is : {(-1)*Ridge_model.best_score_}")

lasso = Lasso()
Lasso_model = GridSearchCV(lasso,parameters,scoring='neg_mean_absolute_error',cv = 5)
Lasso_model.fit(X_train,Y_train)
Y_predicted = Lasso_model.predict(X_test)
print(f"Evaluation matrics MAE for Lasso regression is : {(-1)*Lasso_model.best_score_}")


# print(X.head())
# print(Y.head())