from keras.models import load_model
import h5py
import random
import numpy


def print():
    arr = numpy.random.rand(1,10)
    model = load_model('my_model.h5')
    #print(model.predict(arr))
    return model.predict(arr)
