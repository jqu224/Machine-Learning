import sys
sys.path.append('./')
import pickle
import pandas as pd
import os
if os.path.isfile('titanic.py'):
    from titanic import titanicClassifier
# if os.path.isfile('mnist.py'):
#     from mnist import mnistClassifier


def report_accuracy(dataset_name):
    """
    Test the accuracy of a classifier.
    :param dataset_name: either 'titanic' or 'mnist'
    """
    # load model file
    model_file = open(dataset_name + '_classifier.pkl', 'rb')
    model_wrapper = pickle.load(model_file)

    # model_wrapper is a dict with two values, your nickname and your classifier
    nickname = model_wrapper['nickname']
    model = model_wrapper['classifier']

    # the testing data is saved as the same format as the data given to you, while NO header anymore!!
    # you can randomly copy some lines in the training csv file to form the test data file.
    testing_file = dataset_name + '_test.csv'
    test_df = pd.read_csv(testing_file, header=None)
    test_df = pd.read_csv(testing_file, index_col = False)
    # x_test = test_df.iloc[:, 1:].values
    # y_test = test_df.iloc[:, 0].values
    x_test = test_df.drop(['Survived'], axis=1)
    y_test = test_df['Survived']

    # test classifier accuracy
    accuracy = model.score(x_test, y_test)
    print('%s, your accuracy on ' % nickname + dataset_name + ' test set is %.3f.' % accuracy)

    # Further code to post the accuracy value online.


if __name__ == '__main__':
    report_accuracy(dataset_name='titanic')
