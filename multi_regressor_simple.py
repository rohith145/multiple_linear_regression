import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

dataset=pd.read_csv("breast_cancer.csv")
print(dataset.shape)

dataset = dataset.iloc[:,1:-1]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
dataset.iloc[:,0] = label_encoder.fit_transform(dataset.iloc[:,0]).astype('float64')

x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:,1:].values, dataset.iloc[:,0].values, test_size = 0.2)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



cm = confusion_matrix(y_test, y_pred.round())
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)

