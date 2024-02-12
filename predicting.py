from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
#plt.style.use('seaborn-dark-grid')
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Read the csv file using read_csv
# method of pandas
df = pd.read_csv(r'RELIANCE.csv.csv')
print(df.head())

# Changes The Date column as index columns
df.index = pd.to_datetime(df['Date'])
print(df)

# drop The original date column
df = df.drop(['Date'], axis='columns')
print(df)

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]
X.head()

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
print(y)

split_percentage = 0.7
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
#cls = SVC(kernel= 'rbf', probability= True, C = 1, gamma= 100).fit(X_train, y_train)
cls = SVC().fit(X_train, y_train)
y_pred = cls.predict(X_test)
print("accuracy", accuracy_score(y_test, y_pred) * 100)

df['Predicted_Signal'] = cls.predict(X)
# Calculate daily returns
df['Return'] = df.Close.pct_change()
# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()
print(df)
# Plot Strategy Cumulative returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
print(df)


plt.plot(df['Cum_Return'], color='red',label= 'Actual returns')
plt.plot(df['Cum_Strategy'], color='blue', label='Predicted returns')
plt.xlabel('Time Scale')
plt.ylabel('daily returns(in percentage)')
plt.legend()
plt.show()


plt.style.use('fivethirtyeight')
df1= pd.read_csv(r'RELIANCE.csv.csv')
df1= df1.set_index(pd.DatetimeIndex(df1['Date'].values))

#create SMA
def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()
def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust= False).mean()
def MACD(data, period_long=26,period_short=12,period_signal=9,column='Close'):
    shortEMA = EMA(data,period_short,column=column)
    longEMA=EMA(data,period_long,column=column)
    data['MACD']=shortEMA-longEMA
    data['signal_line']=EMA(data,period_signal,column='MACD')
    return data

def RSI(data,period=14,column ='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] =0
    down[down > 0]=0
    data['up'] = up
    data['down']= down
    AVG_Gain = SMA(data,period,column = 'up')
    AVG_Loss = abs(SMA(data,period,column='down'))
    RS = AVG_Gain/AVG_Loss
    RSI = 100.0 -(100.0/(1.0 + RS))

    data['RSI'] = RSI

    return data

MACD(df1)
RSI(df1)
df1['SMA'] = SMA(df1)
df1['EMA'] = EMA(df1)

column_list = ['MACD','signal_line']
df1[column_list].plot(figsize =(12.2,6.4))
plt.title('MACD ')
plt.ylabel('price')
plt.xlabel('date')
plt.show()

column_list = ['SMA','Close']
df1[column_list].plot(figsize=(12.2,6.4))
plt.title('SMA  ')
plt.ylabel('Price')
plt.xlabel('date')
plt.show()

column_list = ['RSI']
df1[column_list].plot(figsize=(12.2,6.4))
plt.title('RSI ')
plt.ylabel('Price')
plt.xlabel('date')
plt.show()

