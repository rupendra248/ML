import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##Importing dataset from scikit-learn
from sklearn import linear_model
from sklearn.datasets import load_boston
housing_data = load_boston()


##The housing data is in dictionary format, it needs to be converted to pandas dataframe to be used with pandas"
print(housing_data.keys())
housing_df = pd.DataFrame(housing_data.data)


#We can verify the dataframe's top 5 and bottom 5 records
print(housing_df.head())
print(housing_df.tail())


#From the output we see that columns of the dataframe don't have any names yet, so we can assgin the names to columns
# We can get the column names from housing data feature names
print(housing_data.feature_names)
housing_df.columns = housing_data.feature_names
housing_df.info()



#housing_df dataframe has all the independant variables (Xs) now we will add dependant variable Y to the dataframe.
#Our dependant variable is Target (MEDV is median value of owner-occupied homes, in $1,000s)
housing_df['MEDV']= housing_data.target
housing_df.info()


#Next step is to check the distribution of response (Y variable), which is MEDV
sns.distplot(housing_df['MEDV'])
plt.show()


#Looking at the distribution of response variable, it does not look normal and skewed towards the right side
#we can try to either square root transform or log transform.Let's evaluate Log transform
sns.distplot(housing_df['MEDV'].apply(np.log))
plt.show()



#Seperating Y variable (Target = MEDV) and X variables (independant variables)
#After this log transforming the Target variable
X = housing_df.drop('MEDV',axis =1)
#y = housing_df['MEDV']
ylog = housing_df['MEDV'].apply(np.log)
sns.distplot(ylog)
plt.show()



#Next step is to split the data into training and test datasets. Models are trained on the
#training datasets and are tested on test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,ylog_train,ylog_test = train_test_split(X,ylog,test_size = 0.3,random_state = 50)
print(X_train.shape)
print(ylog_train.shape)
print(X_test.shape)
print(ylog_test.shape)




#Next step is to train the model
rdge = linear_model.Ridge(alpha=0.5)
rdge.fit(X_train,ylog_train)
print("Intercept = ",rdge.intercept_)
print(pd.DataFrame(rdge.coef_,X.columns,columns = ['Coefficient']))



# Next step is to test the model using text dataset, this will give us the accuracy with which the model predicts
prediction = rdge.predict(X_test)
#print(prediction)
plt.scatter(ylog_test,prediction)
plt.xlabel("Actual Log Median Price")
plt.ylabel("Predicted Log Median Price")
plt.show()