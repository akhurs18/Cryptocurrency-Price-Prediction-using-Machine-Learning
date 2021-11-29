import numpy as np
import pyspark as spark
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import re
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVR
from sklearn import svm

df = pd.read_csv('volume.csv', error_bad_lines=False)
#df = pd.read_csv('volume.csv', error_bad_lines=False)
df = df[['Date', 'volume']]
df['Date'] = pd.to_datetime(df['Date']).dt.date
print(df)


df2 = pd.read_csv('cryptodata.csv', error_bad_lines=False)
df3 = df2[['time', 'close']]
df3.rename(columns={'time': 'Date'}, inplace=True)

df3['Date'] = pd.to_datetime(df3['Date']).dt.date


merge=pd.merge(df,df3, how='inner', on='Date')
#print(merge)

merge['Price Difference']= merge['close'].diff()
merge.dropna(inplace =True)

rise=1
fall=0
merge['crypto trend']= numpy.where(merge['Price Difference'] > 0 , rise , fall)
#print(merge)
merge['date_delta'] = (merge['Date'] - merge['Date'].min())/ numpy.timedelta64(1,'D')
merge.drop('Date', axis=1, inplace=True)




X = np.array(merge.drop(['crypto trend'],1))


y= merge['crypto trend']


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

svm = svm.SVC(kernel="rbf")

svm.fit(x_train,y_train)
#predict the response
svm_prediction = svm.predict(x_test)


print(classification_report(y_test, svm_prediction))

# lr = svm.SVC(kernel="linear", C= 1e3, gamma= 0.1)

# lr.fit(x_train,y_train)
# #predict the response
# lr_prediction = lr.predict(x_test)


# print(classification_report(y_test, lr_prediction))