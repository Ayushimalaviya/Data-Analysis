#!/usr/bin/env python
# coding: utf-8

# ## Collection & Insight of The Dataset
# 
# CAR QUALITY CLASSIFIACTION/PREDICTION: https://www.kaggle.com/datasets/gagasrock/car-quality-prediction
# 
# This dataset is manually collected from observations. It helps us to build machine learning models to predict the quality of Car whether it is Cheap,Average or High.
# This dataset consists of 10 independent variables i.e Year,Make,Model,Condition,Tranmission,Cylinders,Fuel,Odometer,Engine Power,Milage.
# Generally, the Price Range or Quality of the Car depends on these parameters. These parameters play a vital role in the predictive analysis of the quality of the Car.
# 
# Target: Cheap(0) | Average(1) | High(2)
# 
# The Target Variable is Price which gives the information about the Price range of car whether it is Cheap,Average or High.

# ## Import the required Libraries and load the data into Dataframe

# Data manipulation libraries
import pandas as pd
import numpy as np
import catboost as ctb
import seaborn as sns
import matplotlib.pyplot as plt

#Model development and feature engineering libraries
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve,auc,roc_auc_score
from tensorflow import keras
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df=pd.read_csv("Cars Dataset.csv")


# ## Exploratory Data Analysis
#convert to int type
df.index.astype('int64')
#to check null values
df.isnull().sum()
df.shape

# ### Data Visualization
#to get overview of the data
#df.profile_report()


# ### Which brands are the ones mostly being sold in the second hand market?
ax = sns.countplot(x="MAKE",data=df)
ax.tick_params(axis='x', rotation=90)


# ### Total numbers of Cars with number of cylinders.
sns.countplot(x="CYLINDERS",data=df)
ax.tick_params(axis='x', rotation=90)


# ### Used Cars Market has high demand of cars with Gas fuel.
sns.countplot(x="FUEL",data=df)


# ### Engine Power with Low CC has high demand as compare to Avg CC and High CC
sns.countplot(x="ENGINE POWER",data=df)
sns.countplot(x="Price",data=df)
df.info()

#count model for each car manufacturing company
df.MAKE.value_counts()


# ### How do the Used Car Prices vary as per the Brand

crosstb = pd.crosstab(df.MAKE,df.Price)
crosstb.plot(kind='bar',figsize=(22,10),title='Time taken by Brands to resell');


# ### Is low Engine Power Automatic Car high in demand? 

fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
df.groupby(['ENGINE POWER','TRANSMISSION']).count()['Price'].unstack().plot(ax=ax, title='Total Number of Engine Powered cars sold for each transmission')
plt.ylabel('No. of cars sold') #low engine powered cars sold are automatic

fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['ODOMETER','TRANSMISSION']).count()['MAKE'].unstack().plot(ax=ax, title='Total Distance covered by car for each transmission Type')
plt.ylabel('No. of Car Models') #Around 35 cars covered total distance of almost 80000 are automatic 


# ## Data Preprocessing

# There are few car company with very few records such as 1 or 2 records. So, We dropped those.

#deleted least valued data
df.drop(df[(df['MAKE'] == 'aston-martin') | (df['MAKE'] == 'alfa-romeo') | (df['MAKE'] == 'harley-davidson') | (df['MAKE'] == 'hennessey')].index, inplace=True)
df.reset_index(drop=True, inplace = True)
df.shape

#categorised milage column
def categorize(x):
    if (x >0 and x <= 10):
        return '0'
    if (x > 10 and x <= 20):
        return '1'
    if (x > 20 and x <= 30):
        return '2'
    if x > 30:
        return '3'
#applying the filter function to 'MILAGE' column 
df['category'] = df['MILAGE'].apply(categorize)

df.drop(df[(df['CONDITION'] == 1.0) & (df['Price'] != 0)].index, inplace=True)
df.reset_index(drop=True, inplace = True)
df.shape

df.drop(df[(df['Price'] == 2)].index, inplace=True)
df.reset_index(drop=True, inplace = True)
df.shape

# extracted car age 
df['Current Year'] = 2022
df['YEAR'] = df['Current Year'] - df['YEAR']
df.rename(columns = {'YEAR':'CAR_AGE'}, inplace = True)
df

df.drop(df[(df['CAR_AGE']>20)].index, inplace = True)
df.reset_index(drop=True, inplace = True)
df.shape

# calculated avg odometer to get insights of odometer
df['Avg_odometer'] = df['ODOMETER'].div(df['CAR_AGE']).round(2)

df.drop(df[(df['Avg_odometer']<100) & (df['ODOMETER']<100)].index, inplace = True)
df.reset_index(drop=True, inplace = True)
df.shape


# Also, we have few columns with categorical values, to convert it into numerical(label) using `Label Encoder()`
#encoded the categorical value
Encoder = preprocessing.LabelEncoder()
df['MAKE'] = Encoder.fit_transform(df['MAKE'])
df['FUEL'] = Encoder.fit_transform(df['FUEL'])
df['TRANSMISSION'] = Encoder.fit_transform(df['TRANSMISSION'])
df['CONDITION'] = Encoder.fit_transform(df['CONDITION'])
df['ENGINE POWER'] = Encoder.fit_transform(df['ENGINE POWER'])
df.drop('Current Year', inplace=True, axis =1)

# to get correlation between each feature
x = df.iloc[:,1:10]  
y = df.iloc[:,10]    
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#to get skewed data visualization
df.hist(column =['ODOMETER', 'Avg_odometer','MODEL','CAR_AGE'], grid=False,figsize=(10, 6),bins=30)
df['ODOMETER'].skew()
df.insert(len(df.columns), 'Car_Age', stats.boxcox(df['CAR_AGE'])[0])
df.insert(len(df.columns), 'odometer',stats.boxcox(df['ODOMETER'])[0])

#,'CAR_AGE', 'A_Sqrt'
df.hist(column =['odometer', 'ODOMETER','CAR_AGE','Car_Age'], grid=False,figsize=(10, 6),bins=30)
df.agg(['skew', 'kurtosis']).transpose()

#dropped and renamed unwated and wanted columns
df.drop(['ODOMETER','Avg_odometer','MILAGE', 'CONDITION', 'CAR_AGE','MODEL'], inplace=True, axis=1)
df = df.rename({'odometer': 'ODOMETER','category':'MILEAGE', 'Car_Age' : 'CAR_AGE'}, axis=1)

#convert mileage type to int type
df['MILEAGE']=df['MILEAGE'].astype('int64')
df = df[['CAR_AGE', 'MAKE', 'TRANSMISSION', 'CYLINDERS', 'FUEL', 'ENGINE POWER', 'MILEAGE', 'ODOMETER', 'Price']]

# Spliting the 70% of data into train data & 30% of data into test data
X = df.iloc[:,:8]
y = df.iloc[:,8]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Scaling Continous Values
mms =  MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

sns.pairplot(df, hue='Price', vars=['CAR_AGE','ODOMETER'])

# Feature Selection & Importance
#model development and fitting the features
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#to extract importance of the data
rf.feature_importances_

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='bar')


# Model Development & Execution
# - `Cat Boosting` 
# - `Light Gradient Boosting`
# - `Artificial Neural Network`

# (3) CAT Boosting
#to get cat boost classifier
cbc = ctb.CatBoostClassifier()
#fit the model
cbc = cbc.fit(X_train, y_train)
#predict the model
y_pred = cbc.predict(X_test)


#accuracy, precision, recall and f1 score
print(accuracy_score(y_test, y_pred))
print('Confusion Metrics: \n\n', confusion_matrix(y_test,y_pred))
print('Classification report\n', classification_report(y_test, y_pred))

#heat map visualization
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True)


# ### (4) Light Gradient Boosting (LGB)

#model development
lgb = LGBMClassifier()
model = lgb.fit(X_train,y_train)
lgb_pred = lgb.predict(X_test)
print('Accuracy Score is: ',accuracy_score(y_test, lgb_pred))
print('\n\n\nclassification_report of data after Knn applied: \n',classification_report(y_test, lgb_pred),'\n\n') 
print('Confusion Matrix of data predicted \n',confusion_matrix(y_test, lgb_pred),'\n\n')

sns.heatmap(pd.DataFrame(confusion_matrix(y_test, lgb_pred)), annot=True)


# ## Artificial Neural Network
ann_model_0 = 'Hidden layer 1 = Sigmoid, Hidden_layer 2 =  Relu, Output_Layer: Sigmoid'
ann0 = keras.models.Sequential()
ann0.add(keras.layers.Dense(4, input_dim=8, activation='relu'))
ann0.add(keras.layers.Dense(2, activation='relu'))
ann0.add(keras.layers.Dense(1, activation='relu'))
ann0.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model0 = ann1.fit(X_train, y_train, epochs=80, batch_size = 10)

ann_model_1 = 'Hidden layer 1 = Sigmoid, Hidden_layer 2 =  Sigmoid, Output_Layer: Sigmoid'
ann1 = keras.models.Sequential()
ann1.add(keras.layers.Dense(4, input_dim=8, activation='sigmoid'))
ann1.add(keras.layers.Dense(4, activation='sigmoid'))
ann1.add(keras.layers.Dense(1, activation='sigmoid'))
ann1.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model1 = ann1.fit(X_train, y_train, epochs=80, batch_size = 10)

dic = {ann_model_0:ann0, ann_model_1:ann1}
for i, j in dic.items():
    print("Activation Function applied on",i,'\n\n')
    y_predict = j.predict(X_test)
    y_predict = (y_predict>0.5)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    print("Loss and Accuracy for ANN Model:",j.evaluate(X_test, y_test),'\n\n')
    print('Area Under Curve Score =',roc_auc_score(y_test, y_predict),'\n\n')
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve',linewidth=2)
    plt.ylabel("True Positive Rates")
    plt.xlabel("False Positive Rates")
    plt.title("ROC curve")
    plt.show()
    print('Accuracy Score is: ',accuracy_score(y_test, y_predict))
    print('\n\n\nclassification_report  applied: \n',classification_report(y_test, y_predict),'\n\n') 
    print('Confusion Matrix of data predicted \n',confusion_matrix(y_test, y_predict),'\n\n')

