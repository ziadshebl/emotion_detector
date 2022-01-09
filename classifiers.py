import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import timeit
import sys


class Classifier:
    def __init__(self, type):
        self.type = type
        if (type == "knn"):
            self.clf = KNeighborsClassifier(n_neighbors=5)
        elif(type == "svm"):
            self.clf = svm.SVC(probability=True, verbose = 1) 
        elif(type == "rf"):     
            self.clf = RandomForestClassifier(max_depth=8, random_state=0)
        elif(type == "nn"):
            self.clf = MLPClassifier(hidden_layer_sizes=[100, 50], max_iter=400)
        elif(type == "lda"):
            self.clf = LinearDiscriminantAnalysis()    

    def fit(self, x_train, y_train):
        start = timeit.default_timer()
        self.clf.fit(x_train, y_train)
      
        stop = timeit.default_timer()
        print('Time to fit',self.type,':', stop - start, 'seconds')
        filename = os.path.abspath(os.curdir)+'\\Classifier_Models\\' + self.type +'_model.sav'  
        joblib.dump(self.clf, filename)
        print('Model size of' ,self.type, ':', os.path.getsize(filename), 'bytes')

    def load_model(self):
        filename = os.path.abspath(os.curdir)+'\\Classifier_Models\\'+self.type + '_model.sav'
        self.clf = joblib.load(filename)

    def predict(self, x_test):
        results = self.clf.predict_proba(x_test)
        return np.argmax(results),  np.amax(results)

    def calculate_score(self, y_test, y_predicted):
        correct_count = 0
        for x, y in zip(y_test, y_predicted):
            if(x == y):
                correct_count = correct_count + 1   
        
        
        score = correct_count/len(y_test)
        conf_matrix = confusion_matrix(y_test, y_predicted)
        print(score)
        print(self.type, "confusion matrix")
        print(conf_matrix)
        return score



