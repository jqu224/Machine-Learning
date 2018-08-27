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
import numpy 
import random


print(2)
