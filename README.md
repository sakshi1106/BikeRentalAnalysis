# BikeRentalAnalysis

import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import math
root = tk.Tk()
root.resizable(height = None, width = None)
root.title("Bike Rental Prediction")
global rmse1,rmse2,sub1
global v,v1,v2,v3,v4
global u,u1,u2,u4

def getCSV1():     #method for train dataset
 global train,training,validation
 import_file_path=tk.filedialog.askopenfilename()
 v.set(import_file_path)
 train=pd.read_csv(import_file_path)
 v1.set(train.shape)
 v2.set(train.columns)
 v3.set(train.dtypes)
 train["date"] = pd.DatetimeIndex(train['datetime']).date
 train["hour"] = pd.DatetimeIndex(train['datetime']).hour
 train["month"] = pd.DatetimeIndex(train['datetime']).month
 training = train[train['datetime'] <= '2012-03-30 0:00:00']
 validation = train[train['datetime'] > '2012-03-30 0:00:00']
 train = train.drop(['datetime', 'date', 'atemp'], axis=1)
 training = training.drop(['datetime', 'date', 'atemp'], axis=1)
 validation = validation.drop(['datetime', 'date', 'atemp'], axis=1)

#performing univariate analysis
def Univariate_analysis_count():
  sn.displot(np.log(train["count"]))
  plt.show()

def Univariate_analysis_registered():
 sn.displot(train['registered'])
 plt.show()
def Univariate_analysis_season():
 sn.displot(train['season'])
 plt.show()

#plotting correlation plot
def plot():
 cor = train.corr()
 plt.figure(figsize=(16, 10))
 sn.heatmap(cor)
 plt.show()

#method for test dataset
def getCSV2():
 import_file_path1=tk.filedialog.askopenfilename()
 u.set(import_file_path1)
 test=pd.read_csv(import_file_path1)
 u1.set(test.shape)
 u2.set(test.columns)
 test["date"] = pd.DatetimeIndex(test['datetime']).date
 test["hour"] = pd.DatetimeIndex(test['datetime']).hour
 test["month"] = pd.DatetimeIndex(test['datetime']).month
 test = test.drop(['datetime', 'date', 'atemp'], axis=1)

#calculate rmse value
def rmse(y, y_):
 global rmse_val
 y = np.exp(y),   # taking the exponential as we took the log of target variable\n",
 y_ = np.exp(y_)
 log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
 log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
 calc = (log1 - log2) ** 2
 return np.sqrt(np.mean(calc))

#defining Linear Regression model
def LinearRegressionModel():
 global  X_train,y_train,X_val , prediction
 global y_val
 lModel = LinearRegression()
 X_train = training.drop('count', 1)
 y_train = np.log(training['count'])
 X_val = validation.drop('count', 1)
 y_val = np.log(validation['count'])
 print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
 fitting = lModel.fit(X_train, y_train)  # x is explanatory variable and y is dependent variable
 prediction = lModel.predict(X_val)  # making prediction on validation set
 rmse1.set(rmse(y_val,prediction))
def DecisionTreeRegressor():
  from DesicionReg import rmse21
  rmse2.set(rmse21)
def submission():
 from Submission import sub
 sub1.set(sub)






v=tk.StringVar();v1=tk.IntVar();v2=tk.StringVar();v3=tk.StringVar();rmse2=tk.DoubleVar()
u=tk.StringVar();u1=tk.DoubleVar();u2=tk.StringVar();rmse1=tk.DoubleVar();sub1=tk.StringVar()
entry=tk.Entry(root,textvariable=v1,width='50').grid(row=1,column=2)
tk.Label(text="File selected",bg='green',fg='white',width=10).grid(row=0,column=1)
tk.Label(text="Count value",bg='green',fg='white',width=10).grid(row=0,column=2)
tk.Label(text="Train dataset columns",bg='green',fg='white',width=20).grid(row=3,column=0)
entry=tk.Label(root,textvariable=v2,width=60,height=5).grid(row=3,column=1)
tk.Label(text="Test dataset columns",bg='green',fg='white',width=20).grid(row=4,column=0)
tk.Label(text="Univariate Analysis",bg='green',fg='white',width=20).grid(row=5,column=0)
tk.Button(text="Count",command=Univariate_analysis_count,bg='blue',fg='white').grid(row=6,column=0)
tk.Button(text="Registered",command=Univariate_analysis_registered,bg='red',fg='white').grid(row=7,column=0)
tk.Button(text="Season",command=Univariate_analysis_season,bg='yellow',fg='black').grid(row=8,column=0)
tk.Button(text="Correlation plot",command=plot,bg='pink',fg='black').grid(row=6,column=1)
tk.Label(text="Linear Regression",bg='green',fg='white',width=20).grid(row=6,column=2)
tk.Button(text="Calculate RMSE",command=LinearRegressionModel,bg='pink',fg='black').grid(row=7,column=2)
entry=tk.Label(root,textvariable=u2,width=60,height=5).grid(row=4,column=1)
entry=tk.Label(root,textvariable=rmse1,width='50').grid(row=8,column=2)
tk.Label(text="Decision Tree",bg='green',fg='white',width=20).grid(row=9,column=2)
tk.Button(text="Calculate RMSE",command=DecisionTreeRegressor,bg='pink',fg='black').grid(row=10,column=2)
tk.Button(text="Submission",command=submission,bg='green',fg='white').grid(row=11,column=1)
entry1=tk.Label(root,textvariable=rmse2,width='50').grid(row=11,column=2)
tk.Label(text="Train dataset dtypes",bg='green',fg='white',width=20).grid(row=3,column=2)
entry=tk.Label(root,textvariable=v3,width='50').grid(row=4,column=2)
tk.Button(root,text="Import train dataset",command=getCSV1,bg='pink',fg='black',width='20').grid(row=1,column=0)
train_datasetEntry=tk.Entry(root,textvariable=v,width='50').grid(row=1,column=1)
test_dataButton=tk.Button(text="Import test dataset",command=getCSV2,bg='pink',fg='black',width='20').grid(row=2,column=0)
test_datasetEntry=tk.Entry(root,textvariable=u,width='50').grid(row=2,column=1)
entry=tk.Entry(root,textvariable=u1,width='50').grid(row=2,column=2)







root.mainloop()
