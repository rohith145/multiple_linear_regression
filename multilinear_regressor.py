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


import statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns

SL = 0.05
selected_columns=dataset.columns
selected_columns=selected_columns[1:].values
data_modeled, selected_columns = backwardElimination(dataset.iloc[:,1:].values, dataset.iloc[:,0].values, SL, selected_columns)

result = pd.DataFrame()
result['diagnosis'] = dataset.iloc[:,0]

data = pd.DataFrame(data = data_modeled, columns = selected_columns)
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size = 0.2)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



cm = confusion_matrix(y_test, y_pred.round())
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)
