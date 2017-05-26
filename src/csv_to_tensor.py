import sys
import csv
import numpy
import theano
import theano.tensor as T
from PIL import Image


def load_data(train, targetColumn, targets = 1):
    train_xy = load_csv(train, ',')
    #valid_xy = load_csv_data(valid, ',')
    #test_xy = load_csv_data(test, ',')

    targetColumnNumber = labelToPos(train_xy, targetColumn)

    #test_set_x, test_set_y = transform(test_xy, targetColumnNumber, targets)
    #valid_set_x, valid_set_y = transform(valid_xy, targetColumnNumber, targets)
    train_set_x, train_set_y = transform(train_xy, targetColumnNumber, targets)

    rval = [(train_set_x, train_set_y)]

    return rval

def visual(row):
    img=numpy.reshape(row,(28,28))
    im = Image.fromarray(img)
    im.show()

def transform(data, targetColumnNumber, targets):
    label = []

    del data[0]
    for i in xrange(len(data)):
        label.append(data[i][targetColumnNumber]) #labels: 1 - predictClass, 0 - reg. predict
    data = zip(*data)
    data = map(list, zip(*data[targets:]))

    # conversion to numpy
    data = numpy.array(data, numpy.int32)
   
    label = numpy.array(label, numpy.int32)

    # conversion to theano shared variables
    shared_x = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(numpy.asarray(label, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, 'int32')
    return data,label

def labelToPos(data, label):
    header = {}
    for i in xrange(len(data[0])):
        header[data[0][i]] = i

    try:
        return header[label]
    except KeyError:
        print 'ERROR: Target with name ' + label + ' does not exist'
        sys.exit()

def load_csv(path,delim):
    try:
        csv_file = open(path)
    except:
        print "exception"
        sys.exit(2)
    csv_file = csv.reader(csv_file,delimiter=delim)
    data = [row for row in csv_file]
    return data
