# encoding: utf-8
# !/usr/bin/python
#
# Titanic Dataset: Dead or Alive?
# Project 2.1, COMP 135 Intro to ML
# Tufts ID Jqu03
#
# to build a model that can predict if [Jack] survived the sinking
# based on data about the survivors and deceased.
# Your task is build a model that can predict
# as accurately as possible
# whether a person survive the Titanic tragedy.
#
# given a dataset consist of 800 rows,
# and each row represents a passenger
# that boarded the RMS Titanic.
# Each passenger’s info is made up of 11 features,
# which are:
#
# Survived / Pclass / Sex / Age / sibsp
# / parch / ticket / fare / cabin / embarked / magic_feature

# Goals:
#   The candidate set for each learning algorithm
#   should contain at least four settings.
#
#   checking the performance of the learned classifier
#   through the cross validation procedure.
#

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

    # normalize the following features: ages, fares, ticket and SibSp
    # keep the following features as they were: Pclass, sex, embarked, magic_feature
    # drop the following feature: Parch
    # Global Variables for percentiles of each features
    # that can only be initialized once:

    # step 1: playing with features
    def transform_features(self, images):
        images[images<5] = float('NaN')
        global perc25, perc50, perc75
        if perc25 is None:
            perc25 = np.nanpercentile(images, 25) #100.0
            perc50 = np.nanpercentile(images, 50) #213.0
            perc75 = np.nanpercentile(images, 75) #253.0
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
    labeled_images = pd.read_csv(training_file)
    labeled_images = labeled_images.as_matrix()
    # labeled_images = pd.read_csv(testing_file, index_col = False)

    images = labeled_images[:,1:785]
    labels = np.array(labeled_images[:,0])
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


    # print(perc25,perc50,perc75)

    train_images = mnclf.transform_features(train_images)

    m = 5*np.arange(1,16,2)
    validation_error = []
    for valid_opt in m:
        clfer = AdaBoostClassifier(n_estimators=valid_opt)
        validation_error += [round(100-float(mnclf.run_kfold(clfer,train_images,train_labels)),5)]

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
    # test_labels = np.array(labeled_images[:,0])

    # model_wrapper is a dict with two values, your nickname and your classifier
    nickname = model_wrapper['nickname']
    model = model_wrapper['classifier']
    print(nickname)
    print('%s, your accuracy on' % nickname + ' mnist test set is %.3f.' % model.score(test_images, test_labels))


main()
