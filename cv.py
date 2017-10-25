import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import os
import csv
with open('best_params.csv','w') as write_csv:
    writer = csv.writer(write_csv, delimiter=',')
    for file in os.listdir(os.getcwd()+"/Image-data (1)"):
        if file.endswith(".csv"):
            balance_data = pd.read_csv(os.getcwd()+"/Image-data (1)/"+str(file), sep= ',')

            #print "Dataset Lenght:: ", len(balance_data)
            #print "Dataset Shape:: ", balance_data.shape

            X = balance_data.values[:,1:12].astype(float)
            Y = balance_data.values[:,12].astype(float)


            X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

            param_grid={ 
                        'n_estimators':range(100,1200,300), 
                        'learning_rate': [0.1,0.05,0.02,0.04],
                        'max_depth':range(5,15,2),
                        'min_samples_split': range(500,1000,200),
                        'min_samples_leaf': range(50,100,20),
                        'max_features':[4, 5, 6] 
                        }

            print("# Tuning hyper-parameters: " + str(file))
            print()

            clf = GridSearchCV(ensemble.GradientBoostingRegressor(learning_rate=0.2, n_estimators=70, min_samples_split=600, min_samples_leaf=50, max_depth=8, subsample=0.8), param_grid, cv=5, n_jobs=5)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            writer.writerow([file,clf.best_params_])
            



