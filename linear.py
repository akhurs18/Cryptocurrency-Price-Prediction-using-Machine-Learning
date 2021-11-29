import numpy as np
import pyspark as spark
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.metrics import precision_recall_curve


df = pd.read_csv('test2.csv', error_bad_lines=False)
#print(df)


df2 = pd.read_csv('cryptodata.csv', error_bad_lines=False)
df3 = df2[['time', 'close']]
df3.rename(columns={'time': 'Date'}, inplace=True)

#print(df2)

df['Date']= pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
df3['Date']= pd.to_datetime(df3['Date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
#df['Date'] = df['Date'].astype('datetime64[ns]')
#print(df3)



df['Date'] = df['Date'].dt.floor('h')
print(df)

merge=pd.concat([df, df3])

#merge=pd.merge(df,df3, how='inner', on='Date')
#print(merge)


mean1=merge.groupby(['Date'], as_index=False).mean()
#print(mean1)
mean1['Price Difference']= mean1['close'].diff()
mean1.dropna(inplace =True)


rise=1
fall=0
mean1['crypto trend']= numpy.where(mean1['Price Difference'] > 0 , rise , fall)
#print(mean1)
mean1['date_delta'] = (mean1['Date'] - mean1['Date'].min())/ numpy.timedelta64(1,'D')
mean1.drop('Date', axis=1, inplace=True)
#mean1.drop('Unnamed: 0', axis=1, inplace=True)

X = np.array(mean1.drop(['crypto trend'],1))


y= mean1['crypto trend']
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


lr = LinearDiscriminantAnalysis()

lr.fit(x_train,y_train)
#predict the response
lr_prediction = lr.predict(x_test)

print(lr_prediction)
#print(y_test)
print(classification_report(y_test, lr_prediction))



precision, recall, thresholds = precision_recall_curve(y_test, lr_prediction)



