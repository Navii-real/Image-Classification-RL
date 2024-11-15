from numpy import array
import numpy as np
from getData import getData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report, make_scorer,accuracy_score



def doSVM(features:array, labels: array):
    pca = PCA(n_components=20)
    scl = StandardScaler()
    features = scl.fit_transform(features)
    features = pca.fit_transform(features)
    clf = svm.SVC()
    clf.fit(features,labels)
    return clf

def testSVM(features:array, labels: array, ytest: array, xtest: array):
    clf = doSVM(features, labels)
    pca = PCA(n_components=20)
    scl = StandardScaler()
    features = scl.fit_transform(features)
    features = pca.fit_transform(features)
    xtest = scl.fit_transform(xtest)
    xtest = pca.fit_transform(xtest)
    predictedSVM = clf.predict(xtest)
    print(clf.score(xtest, ytest))
    #setup to get f-score and cv
    #scorerVar = make_scorer(f1_score, pos_label=1)
    #f1_scores = cross_val_score(clf, features, labels, scoring = scorerVar, cv = 5)
    #print(f1_scores)
    #confusion matrix
    print(confusion_matrix(ytest, predictedSVM))
    #classification report
    print(classification_report(ytest, predictedSVM, labels = [1]))
    print(accuracy_score(ytest,predictedSVM))

if __name__ == "__main__":
    xTrain,yTrain = getData.loadFeatureMat()
    xTest,yTest = getData.loadFeatureMat(dataset="test")
    testSVM(xTrain,yTrain,yTest,xTest)
