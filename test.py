
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# For data manipulation
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Read the csv file using read_csv
# method of pandas
df = pd.read_csv(r'RELIANCE.csv.csv')
#print(df.head())

# Changes The Date column as index columns
df.index = pd.to_datetime(df['Date'])
#print(df)

# drop The original date column
df = df.drop(['Date'], axis='columns')
#print(df)

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]
X.head()
scaler = StandardScaler()
#fit n transform
scaled_data = scaler.fit_transform(X)
#print(scaled_data)
# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3,stratify=y,random_state=2) # 70% training and 30% test


# Support vector classifier
cls = SVC(kernel= 'rbf', probability= True, C= 0.00011, gamma= 600).fit(X_train, y_train)
#cls = SVC(kernel= 'rbf').fit(X_train, y_train)

y_pred = cls.predict(X_test)
#model accuracy
print("accuracy", accuracy_score(y_test, y_pred) * 100)
print('confusion matrix')
results = confusion_matrix(y_test,y_test)
print(results)
print('classification report')
print(classification_report(y_test,y_pred))