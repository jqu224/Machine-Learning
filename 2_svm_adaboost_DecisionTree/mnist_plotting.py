# encoding: utf-8
# !/usr/bin/python
#
# MNIST Dataset: 8 or 9?
# Project 2.1, COMP 135 Intro to ML
# Tufts ID Jqu03
#
# to build a model that can predict if [?] is 8 or 9
# based on images of [labeled] digits.
# Your task is build a model that can predict
# as accurately as possible
#
# given a dataset consist of 8800 rows,
# and each row represents a digit either 8 or 9
# Each digit consists of 784 pixels
#
# Step 1:
# normalize the pixels convert 0 - 255 to 0 - 4
# Step 2:
# modeling, validation and testing
# the three classifiers being used here are svm, decision tree and adaBoost

import nltk
import string
import glob
import sys
import random
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import math
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import tree, svm, naive_bayes
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier



perc25 = None
perc50 = None
perc75 = None
class mnistClassifier():
    clfier = None
    def __init__(self):
        pass

    def clfclf(self, clf):
        clfier = clf
        return clfier

    # step 1: playing with features
    def transform_features(self, images):
        images[images<5] = float('NaN')
        global perc25, perc50, perc75
        if perc25 is None:
            # find the q1 q2 q3's values
            perc25 = np.nanpercentile(images, 25) # 100.0
            perc50 = np.nanpercentile(images, 50) # 213.0
            perc75 = np.nanpercentile(images, 75) # 253.0
        images[np.isnan(images)] = int(0)
        images[images<perc25] = int(1)
        images[np.where((images<perc50) & (images>=perc25))] = int(2)
        images[np.where((images>perc50) & (images<=perc75))] = int(3)
        images[images>=perc75] = int(4)
        return images

    # k-fold validation
    def run_kfold(self,clf,X_train,y_train):
        kfold = KFold(n_splits=5)
        model = clf
        results = cross_val_score(model, X_train, y_train, cv=kfold)
        return results.mean() * 100.0

    # main function
    def score(self, X, y, **kwargs):
    # include all your data pre-processing
        X = self.transform_features(X)
        clf = self.clfier
        y_pred = clf.predict(X)
    # call the score() function in the sklearn
        return accuracy_score(y, y_pred)

def main():
    mnclf = mnistClassifier()

    training_file = 'mnist_binary.csv'
    # labeled_images = pd.read_csv(training_file, header=None)
    labeled_images = pd.read_csv(training_file, index_col = False)
    labeled_images = labeled_images.as_matrix()

    images = np.array(labeled_images[:,1:785])
    labels = np.array(labeled_images[:,:1])
    num_test = 0.2
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=num_test, random_state=23)
    # train_images = mnclf.transform_features(train_images)

    plt.figure(1)
    ada_m = 8*np.arange(1,16,3)
    validation_error = []
    for valid_opt in ada_m:
        clfer = AdaBoostClassifier(n_estimators=valid_opt)
        clfer.fit(train_images, train_labels)
        validation_error += [round(100-float(mnclf.run_kfold(clfer,train_images,train_labels)),5)]
    plt.title('adaboost with different n_estimators')
    width = 1/1.5
    y_pos = np.arange(len(ada_m))
    plt.xticks(y_pos, ada_m)
    plt.bar(y_pos, validation_error, align='center', alpha=0.5)
    plt.ylabel('validation_error rate %')

    plt.figure(2)
    sv_m = [0.001, 0.1, 2.5, 12.5, 200, 2000]
    for valid_opt in sv_m:
        clfer = svm.SVC(C = valid_opt)
        validation_error += [round(100-float(mnclf.run_kfold(clfer,train_images,train_labels)),5)]

    plt.title('svm with different values of C ')
    width = 1/1.5
    y_pos = np.arange(len(sv_m))
    plt.xticks(y_pos, sv_m)
    plt.bar(y_pos, validation_error, align='center', alpha=0.5)
    plt.ylabel('validation_error rate %')

    plt.figure(3)
    dc_m = [1,3,5,7,8]
    for valid_opt in dc_m:
        clfer = tree.DecisionTreeClassifier(max_features = valid_opt)
        validation_error += [round(100-float(mnclf.run_kfold(clfer,train_images,train_labels)),5)]

    plt.title('Decision tree with different max_features')
    width = 1/1.5
    y_pos = np.arange(len(dc_m))
    plt.xticks(y_pos, dc_m)
    plt.bar(y_pos, validation_error, align='center', alpha=0.5)
    plt.ylabel('validation_error rate %')

    plt.show()



    # choose the third best setting
    final_parameter = sorted(validation_error)[2]
    final_choice = validation_error.index(final_parameter)
    final_choice = m[final_choice]
    print('error rate of my final_choice of # of estimators is: ', final_choice)
    final_clfer = AdaBoostClassifier(n_estimators=final_choice)
    final_clfer.fit(train_images, train_labels)
    mnclf.clfier = final_clfer
    classifier = {'nickname': 'catfacebig', 'classifier': mnclf}
    pickle.dump(classifier, open('mnist_classifier.pkl', 'wb'))

    model_file = open('mnist_classifier.pkl', 'rb')
    model_wrapper = pickle.load(model_file)


    # # # # # # # # # # #
    # # # for testing # #
    # # # # # # # # # # #
    # testing_file = 'mnist_test.csv'
    # labeled_images = pd.read_csv(testing_file)
    # labeled_images = labeled_images.as_matrix()
    # # test_df = pd.read_csv(labeled_images,index_col = False)
    #
    # test_images = labeled_images[:,1:785]
    # test_labels = np.array(labeled_images[:,:1])

    # model_wrapper is a dict with two values, your nickname and your classifier
    nickname = model_wrapper['nickname']
    model = model_wrapper['classifier']
    print(nickname)
    print('%s, your accuracy on' % nickname + ' mnist test set is %.3f.' % model.score(test_images, test_labels))


main()
