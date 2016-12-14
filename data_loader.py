import gzip
import cPickle
import numpy as np


def get_data():
    data = gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data = cPickle.load(data)
    data.close()
    training_data,test_data = format_data(training_data,test_data)
    return training_data,test_data

def format_data(training_data,test_data):
    training_in = [np.reshape(i, (784,1)) for i in training_data[0]]
    training_out = [vectorize(i) for i in training_data[1]]
    training_data = zip(training_in,training_out)
    test_in =[np.reshape(i, (784,1)) for i in test_data[0]]
    test_data = zip(test_in,test_data[1])
    return(training_data,test_data)
       


def vectorize(x):
    vector = np.zeros((10,1))
    vector[x] = 1.0
    return vector     
