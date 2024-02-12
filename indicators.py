import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
df1= pd.read_csv(r'RELIANCE.NS.csv')
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
plt.axhline(y=70,color='green',linestyle='dashed', lw=2,label='overbought')
plt.axhline(y=30,color='r',linestyle='dashed', lw=2, label='oversold')
plt.title('RSI ')
plt.ylabel('Price')
plt.xlabel('date')
plt.legend(loc="best")
plt.show()
