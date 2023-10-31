#%%
import numpy as np
import pandas as pd
import quandl
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('INFY.csv')
dates = list(range(0,int(len(data))))
prices = data['Close']
data.head()
data.info()
data.describe()
data.columns
df = pd.DataFrame(data, columns=['Date','Close'])

df = df.reset_index()
print(df.head())
df.info()
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.20)
from sklearn.linear_model import LinearRegression
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']
model = LinearRegression()
model.fit(X_train, y_train)
print('Slope: ', np.ndarray.item(np.squeeze(model.coef_)))
if np.ndarray.item(np.squeeze(model.coef_)) > 0.5:
    print("The stock is in uptrend and you can buy the share")

elif np.ndarray.item(np.squeeze(model.coef_)) < -0.5:
    print('stock is in downtrend you can exit your buy position or short sell ')
else:
    print('sideways market ')

print('invest as per your analysis')
print('Intercept: ', model.intercept_)
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Best fit line linear regression')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']
y_pred = model.predict(X_test)
df.shape
randints = np.random.randint(2550, size=25)

df_sample = df[df.index.isin(randints)]
print(df_sample.head())
from scipy.stats import norm

mu, std = norm.fit(y_test - y_pred)

ax = sns.distplot((y_test - y_pred), label='Residual Histogram & Distribution')
        
x = np.linspace(min(y_test - y_pred), max(y_test - y_pred), 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, 'r', lw=2, label='Normal Distribution') 

plt.legend()
plt.show()
df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))
df.head()
data = pd.read_csv('INFY.csv')
data.head(4)
data.info()
data.describe()
X = data[['High','Low','Open','Volume']].values
y = data['Close'].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
Model = LinearRegression()
Model.fit(X_train, y_train)
print(Model.coef_)
predicted = Model.predict(X_test) 
print(predicted)
data1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted' : predicted.flatten()})
print(data1.head(20))
print(data1.head(-1))
import math
from sklearn import preprocessing
from sklearn import metrics
graph = data1.head(20)
graph.plot(kind='bar')
plt.plot(X_test, y_test, color='green',linewidth=1) 
plt.plot(X_test, Model.predict(X_test), color='blue', linewidth=3)
plt.title('Linear Regression | Time vs. Price ')
plt.legend()
plt.xlabel('Date Integer')
plt.show()
