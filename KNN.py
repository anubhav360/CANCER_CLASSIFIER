# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection ,neighbors,svm

df= pd.read_csv('breast-cancer-wisconsin.data.txt');
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
X= np.array(df.drop(['class'],1));
Y=np.array(df['class']);
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.2);
clf=neighbors.KNeighborsClassifier();
clf.fit(X_train,Y_train);
accuracy=clf.score(X_test,Y_test);
print (accuracy);
pre=np.array([4,2,1,1,1,2,3,2,1]);
pre=pre.reshape(1,-1);
f=clf.predict(pre);
print(f);