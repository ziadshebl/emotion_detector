from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np

class Classifier:
    def __init__(self, type):
        self.type = type
        if (type == "knn"):
            self.clf = KNeighborsClassifier(n_neighbors=5)
        elif(type == "svm"):
            self.clf = svm.SVC(probability=True) 
        elif(type == "rf"):     
            self.clf = RandomForestClassifier(max_depth=8, random_state=0)
        elif(type == "nn"):
            self.clf = MLPClassifier(hidden_layer_sizes=[100, 50], max_iter=400)  

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        filename = self.type +'_model.sav'
        joblib.dump(self.clf, filename)

    def load_model(self):
        filename = self.type + '_model.sav'
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
        print(score)
        return score



