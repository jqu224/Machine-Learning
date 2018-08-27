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
# step 1: preprocessing
# normalize the following features: ages, fares, ticket and SibSp
# keep the following features as they were: Pclass, sex, embarked, magic_feature
# dump the following feature: Parch


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
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import tree, svm, naive_bayes
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


Age_5pct = None
Age_30pct = None
Age_60pct = None
Age_80pct = None
Age_100pct = None

Fare_25pct = None
Fare_50pct = None
Fare_75pct = None
Fare_100pct = None

Ticket_20pct = None
Ticket_70pct = None
Ticket_90pct = None
Ticket_100pct = None

SibSp_endpercentile = None

class titanicClassifier():
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
    def simplify_ages(self,df):
        df.Age = df.Age.fillna(-0.5)
        global Age_5pct, Age_30pct, Age_60pct, Age_80pct, Age_100pct
        if Age_5pct is None:
            Age_5pct = np.percentile(df.Age, 5, axis=0)
            Age_30pct = np.percentile(df.Age, 30, axis=0)
            Age_60pct = np.percentile(df.Age, 60, axis=0)
            Age_80pct = np.percentile(df.Age, 80, axis=0)
            Age_100pct = np.percentile(df.Age, 100, axis=0)
        bins = (-1, 0, Age_5pct, Age_30pct, Age_60pct, Age_80pct, Age_100pct)
        group_names = ['Unknown', 'Q1','Q2', 'Q3', 'Q4', 'Q5']
        categories = pd.cut(df.Age, bins, labels=group_names)
        df.Age = categories
        return df

    def simplify_fares(self,df):
        df.Fare = df.Fare.fillna(-0.5)
        global Fare_25pct, Fare_50pct, Fare_75pct, Fare_100pct
        if Fare_25pct is None:
            Fare_25pct = np.percentile(df.Fare, 25, axis=0)
            Fare_50pct = np.percentile(df.Fare, 50, axis=0)
            Fare_75pct = np.percentile(df.Fare, 75, axis=0)
            Fare_100pct = np.percentile(df.Fare, 100, axis=0)
        bins = (-1, 0, Fare_25pct, Fare_50pct, Fare_75pct, Fare_100pct)
        group_names = ['Unknown', 'Q1', 'Q2', 'Q3', 'Q4']
        categories = pd.cut(df.Fare, bins, labels=group_names)
        df.Fare = categories
        return df

    def simplify_ticket(self,df):
        df.Ticket = df.Ticket.fillna(-0.5)
        global Ticket_20pct, Ticket_70pct, Ticket_90pct, Ticket_100pct
        if Ticket_20pct is None:
            Ticket_20pct = np.percentile(df.Ticket, 20, axis=0)
            Ticket_70pct = np.percentile(df.Ticket, 70, axis=0)
            Ticket_90pct = np.percentile(df.Ticket, 90, axis=0)
            Ticket_100pct = np.percentile(df.Ticket, 100, axis=0)
        bins = (-1, 0, Ticket_20pct, Ticket_70pct, Ticket_90pct, Ticket_100pct)
        group_names = ['Unknown', 'Q1', 'Q2', 'Q3', 'Q4']
        categories = pd.cut(df.Ticket, bins, labels=group_names)
        df.Ticket = categories
        return df

    def simplify_SibSp(self,df):
        df.SibSp = df.SibSp.fillna(-0.5)
        global SibSp_endpercentile
        if SibSp_endpercentile is None:
            SibSp_endpercentile = np.percentile(df.SibSp, 100, axis=0)
        bins = (-1, 0.5, 1.5, SibSp_endpercentile)
        group_names = ['0', '1', '> 1']
        categories = pd.cut(df.SibSp, bins, labels=group_names)
        df.SibSp = categories
        return df

    def drop_features(self,df):
        return df.drop(['Parch'], axis=1)

    # step 1: playing with features
    def transform_features(self, df):
        df = self.simplify_fares(df)
        df = self.simplify_ages(df)
        df = self.simplify_ticket(df)
        df = self.simplify_SibSp(df)
        df = self.drop_features(df)
        return df

    # step 2: normalize features
    def encode_features(self,df):
        features = ['Fare', 'Age', 'Ticket', 'SibSp']
        df_combined = pd.concat([df[features]])

        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(df_combined[feature])
            df[feature] = le.transform(df[feature])
        return df

    # k-fold validation
    def run_kfold(self,clf,X_train,y_train):
        kfold = KFold(n_splits=5)
        model = clf
        results = cross_val_score(model, X_train, y_train, cv=kfold)
        return results.mean() * 100.0
        # print(results.std()*100.0)

    # main function
    def score(self, X, y, **kwargs):
    # include all your data pre-processing
        X = self.transform_features(X)
        X = self.encode_features(X)
        clf = self.clfier 
        y_pred = clf.predict(X)
    # call the score() function in the sklearn
        return accuracy_score(y, y_pred)

def main():
    tclf = titanicClassifier()

    data_train = pd.read_csv('titanic_filled.csv')
    X_all = data_train.drop(['Survived'], axis=1)
    y_all = data_train['Survived']

    num_test = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

    X_train = tclf.transform_features(X_train)
    X_train = tclf.encode_features(X_train)
    m = 5*np.arange(1,16,2)
    validation_error = []
    for valid_opt in m:
        clfer = AdaBoostClassifier(n_estimators=valid_opt)
        validation_error += [round(100-float(tclf.run_kfold(clfer,X_train,y_train)),5)]

    # choose the third best setting
    # final_parameter = np.percentile(validation_error, 25, axis=0)
    final_parameter = sorted(validation_error)[2]
    final_choice = validation_error.index(final_parameter)
    final_choice = m[final_choice]
    print('error rate of my final_choice of # of estimators is: ', final_choice)
    final_clfer = AdaBoostClassifier(n_estimators=final_choice)
    final_clfer.fit(X_train, y_train)
    tclf.clfier = final_clfer
    classifier = {'nickname': 'catfacebig', 'classifier': tclf}
    pickle.dump(classifier, open('titanic_classifier.pkl', 'wb'))

    model_file = open('titanic_classifier.pkl', 'rb')
    model_wrapper = pickle.load(model_file)
    df = []
    # # # # # # # # # # # #
    # # # # for testing # #
    # # # # # # # # # # # #
    # testing_file = 'titanic_test.csv'
    # test_df = pd.read_csv(testing_file, header=None)
    # # test_df = pd.read_csv(testing_file,index_col = False)
    #
    # print(test_df.head())
    #
    # x_test = data_train.drop(['Survived'], axis=1)
    # y_test = data_train['Survived']
    #
    # # model_wrapper is a dict with two values, your nickname and your classifier
    # nickname = model_wrapper['nickname']
    # model = model_wrapper['classifier']
    # print(nickname)
    # print('%s, your accuracy on' % nickname + ' titanic test set is %.3f.' % model.score(X_test, y_test))


main()

