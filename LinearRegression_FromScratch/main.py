import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from xgboost import XGBRegressor
check = True

class Linear_Regression():
  
  loss_function_old = 0

  # initiating the parameters (learning rate & no. of iterations)
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations


  def fit(self, X, Y ):

    # number of training examples & number of features

    self.m, self.n = X.shape  # number of rows & columns

    # initiating the weight and bias 

    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y
    
    self.check = True
    # implementing Gradient Descent
    
    for i in range(self.no_of_iterations):
      self.update_weights()
    #   print(f"weight is {self.w}")
    #   print(f"bias is {self.b}")


  def update_weights(self):

    

    Y_prediction = self.predict(self.X)

    # calculate gradients

    dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = - 2 * np.sum(self.Y - Y_prediction)/self.m

    # upadating the weights
    
    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db

    

    loss_function_new = np.sum((self.Y - Y_prediction)**2)/self.m

    # print(f"old loss function value is : {self.__class__.loss_function_old} and new value is : {loss_function_new}")
    
    # print(f"weight gradient is {dw} and bias gradient is {db}")
    
    # print(f"Weight is {self.w} and bias is {self.b}")
    # print("------done-------")

    self.__class__.loss_function_old = loss_function_new
    
    
 

  def predict(self, X):

    return X.dot(self.w) + self.b


salary_data = pd.read_csv('salary_data.csv')

X = salary_data.iloc[:,:-1].values      
Y = salary_data.iloc[:,1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 2)

model = Linear_Regression(learning_rate = 0.02, no_of_iterations=1000)

model.fit(X_train, Y_train)

# print('weight = ', model.w[0])
# print('bias = ', model.b)

Y_predicted = model.predict(X_test)

score = metrics.mean_absolute_error(Y_test,Y_predicted)

print(f"my model mean absolute error is : {score} Rupees")

model2 = XGBRegressor()
model2.fit(X_train,Y_train)
Y_predicted = model2.predict(X_test)
score = metrics.mean_absolute_error(Y_test,Y_predicted) 

print(f"XGBregressor model mean absolute error is : {score} Rupees")



