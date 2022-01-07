import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.svm import SVC

svc = sklearn.svm.SVC()
spam = pd.read_csv('Spam.csv')
spam.isnull().sum()
spam.head()
X=spam.drop('spam',axis=1)
y=spam['spam']

#We can scale the X parameters
# scaling the features
# note that the scale function standardises each column, s.e.
# x1 = x1-mean(x1)/std(x1)

from sklearn.preprocessing import scale
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
svc.score(X_test,y_test)
confusion_matrix(y_test,pred)

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
print("Grid Best Estimator\n",grid.best_estimator_)
print("Grid Best Score\n",grid.best_score_)
grid_predict = grid.predict(X_test)
print(grid.score(X_test,y_test))

