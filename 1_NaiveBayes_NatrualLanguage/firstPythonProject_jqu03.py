# encoding: utf-8
#!/usr/bin/python

import nltk  
import string
import glob
import sys
import numpy 
import random

nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


import os
import scipy
from nltk.stem.porter import PorterStemmer
from sklearn.utils import shuffle

# mapping the constant
POSITIVE = 1
NEGATIVE = 0

def read_in_data_function(filename, d_vocabulary):
    # list of text documents
    messages = [line.rstrip() for line in open(filename)]
    # create the transform
    label = []
    for sentence in messages:
        if sentence[-1] is '1':
            label += [1]
        elif sentence[-1] is '0':
            label += [0]
        else:
            sys.exit("It failed! invalid data file: doesnt have a label in the end of the line")        
    stemmer = nltk.stem.PorterStemmer()
    
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text):
        text = text.replace('/', ' ')
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        translator = str.maketrans('', '', string.digits)
        text = text.translate(translator)
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    vectorizer = CountVectorizer(max_features=d_vocabulary, max_df=.9, tokenizer=tokenize, binary=True, stop_words='english')
    # max_features=1000, max_features=d_vocabulary, max_df=.85, min_df=.002,

    vectorizer.fit(messages) # tokenize and build vocab
    vector = vectorizer.transform(messages) # encode document
    X = vector.toarray() # summarize
    X = numpy.array(X) # convert to numpy
    return X, label 

def split_data(X, Y, train_test_ratio, random_i):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_test_ratio, random_state = random_i)
    return X_train , Y_train , X_test , Y_test
 
def train(X_train , y_train , train_opt):
    len_row,len_col = X_train.shape
    d = len_col
    m = train_opt
    X_pos = numpy.zeros((len_col,), dtype=float)
    X_neg = numpy.zeros((len_col,), dtype=float)
    nPos = sum(y_train)
    nNeg = len(y_train) - nPos
    
    y_train = numpy.array(y_train)
    Y_vector = y_train[numpy.newaxis, :].T
    new_X_train = numpy.append(X_train,Y_vector, axis=1) # put x and y together as [x,y]
    new_X_train = new_X_train[numpy.argsort(new_X_train[:, -1])] # sort by Y
    new_X_train = new_X_train[:,:-1] # take out the Y
    
    X_neg += sum(new_X_train[:nNeg,:])
    X_pos += sum(new_X_train[nNeg:,:])
    
    X_neg0 = (X_neg + m) / (nNeg + m*d)
    X_neg2 = (nNeg - X_neg + m) / (nNeg + m*d)
    X_pos1 = (X_pos + m) / (nPos + m*d)
    X_pos3 = (nPos - X_pos + m) / (nPos + m*d)
    trained_model = numpy.array([X_neg0,X_pos1,X_neg2,X_pos3])
    p_y_1 = nPos/len(y_train) # prior_of_y_1
    prior_vector = numpy.array([[1-p_y_1],[p_y_1],[1-p_y_1],[p_y_1]])
    
    trained_model = numpy.append(trained_model,prior_vector, axis=1)
    return trained_model


def test(X_test, trained_model, Y_test):
    nTest, nFeature = X_test.shape
    y_pred = numpy.array([])
    p_y_0 = trained_model[0,-1] 
    p_y_1 = trained_model[1,-1] 
    trained_model = trained_model[:,:-1] 
    for j in range(nTest):
        pred_1 = numpy.log(p_y_1) + numpy.dot(numpy.log(trained_model[POSITIVE]), X_test[j]) + \
            numpy.dot(numpy.log(trained_model[POSITIVE+2]), numpy.absolute(X_test[j]-1)) 
        pred_0 = numpy.log(p_y_0) + numpy.dot(numpy.log(trained_model[NEGATIVE]), X_test[j]) + \
            numpy.dot(numpy.log(trained_model[POSITIVE+2]), numpy.absolute(X_test[j]-1)) 
        if pred_1 == pred_0:
            flip = random.choice([True, False]) # for the 50/50 boundary cases
#            print("found you hahahaha", Y_test[j])
            pred = 1 if flip == True else 0
        else:
            pred = 1 if (pred_1>pred_0) else 0
        y_pred = numpy.append(y_pred, pred)
    return y_pred 

def evaluate(y_test, y_pred):
    nTest = len(y_test)
    x =numpy.array([y_test,y_pred], dtype=float) 
    error_rate = float(numpy.sum(numpy.absolute(numpy.diff(x, axis=0))) / nTest)
    return error_rate

def main ():
    
    filenames = glob.glob('*.txt') # read in txt files in current directory
    print()
    print("==============================================")
    for idx, file in enumerate(filenames):
        print("#\tfile", idx, "is", file)
    print("==============================================")
    
    filename_idx = int(input("Enter the file's index: <0, 1, 2,...>: \n"))
    if filename_idx < len(filenames):
        datafile_name = filenames[filename_idx]
        print("you've chosen: ", datafile_name)
    else:
        return 0
            
    average = []
    error_rate = []
    train_test_ratio = 100
    vocabulary = 1000
    X, Y = read_in_data_function(datafile_name, vocabulary)

    m = [0.1, 0.5, 2.5, 12.5]
    for train_opt in m:
        for i_seed in range(100):
            X_train , Y_train , X_test , Y_test = split_data(X, Y, train_test_ratio, i_seed)
            trained_model = train(X_train , Y_train, train_opt)
            Y_pred = test(X_test, trained_model, Y_test)
            error_rate += [float(evaluate(Y_test, Y_pred))]
        average += [numpy.mean(error_rate)]
        
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    
    plt.figure(1)
#    plt.subplot(211)
    plt.title('smooth_param')
    plt.plot(m,average)
    plt.ylabel('error rate %%')
    figName = [datafile_name+'_m.png']
    plt.savefig('jqu03_m.png')

    average = []
    error_rate = []
    train_opt = .1
    vocabulary = [1000, 500, 250, 125]
    for d in vocabulary:
        for i_seed in range(10):
            X, Y = read_in_data_function(datafile_name, d)
            X_train , Y_train , X_test , Y_test = split_data(X, Y, train_test_ratio, i_seed)
            trained_model = train(X_train , Y_train, train_opt)
            Y_pred = test(X_test, trained_model, Y_test)
            error_rate += [float(evaluate(Y_test, Y_pred))]
        average += [numpy.mean(error_rate)]
        
        
    print(vocabulary,average)
    plt.figure(2)
#    plt.subplot(212)
    plt.plot(vocabulary,average)
    plt.ylabel('error rate %%')
    plt.xlabel('vocabulary size')
    figName = [datafile_name+'_d.png']
#    plt.grid(True)
    plt.savefig('jqu03_d.png')
    
main()
