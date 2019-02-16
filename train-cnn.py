import sys
import numpy as np
import pprint
import time
import os
import argparse as argparse
import json
import hvutils as hv
import threading
import queue

from tqdm import tqdm
from random import shuffle

from keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.utils import np_utils
from keras.callbacks import History
from keras.initializers import glorot_uniform


def makeModel(input_shape = (100,100,1), classes = 10):

    #hv.set_tf_session_for_keras(memory_fraction=1.0)

    X_input = Input(input_shape)

    #X = ZeroPadding2D((3,3))(X_input)

    X = Dropout(0.01)(X_input)

    X = Conv2D(filters=3, kernel_size= (3,3), strides = (1,1), name='conv1', padding='valid', kernel_initializer='glorot_uniform')(X)
    #X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)

    X = Conv2D(filters=20, kernel_size=(5,5), strides = (1,1), name='conv2', padding='valid', kernel_initializer='glorot_uniform')(X)
    #X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(5,5), strides=(1,1))(X)

    X = Conv2D(filters=30, kernel_size=(7,7), strides = (1,1), name='conv3', padding='valid', kernel_initializer='glorot_uniform')(X)
    #X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(2,2), strides=(1,1))(X)

    # X = Conv2D(filters=512, kernel_size=(9,9), strides = (1,1), name='conv4', padding='valid', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)
    # X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(2,2), strides=(1,1))(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = X, name = '4layers')

    return model

    
def testModel(model, num_dev_files, dev_files_list, classes, mini_batch_size, imgdim):

    num_mini_batches = num_dev_files // mini_batch_size
    loss,acc = None, None

    for bno in tqdm(range(0,num_mini_batches)):
            
        X_test_orig, Y_test_orig = hv.load_minibatch(classes, dev_files_list, mini_batch_size, bno, imgdim)

        X_test = X_test_orig/255
        _, Y_test = hv.convert_to_one_hot(classes, Y_test_orig)

        loss,acc = model.test_on_batch(X_test, Y_test)
    
    print ("Loss [{}] Acc [{}] ".format(str(loss),str(acc)))

batch_data = queue.Queue()


def minibatchLoader(classes, train_files_list, num_mini_batches, mini_batch_size, bno, imgdim):

    while bno < num_mini_batches:

        while batch_data.qsize() > 50:
            if threading.main_thread().is_alive() is False:
                print("Main thread exited, data loader exiting")
                return
            #print("sleeping")
            time.sleep(0.1)

        X_train_orig, Y_train_orig = hv.load_minibatch(classes, train_files_list, mini_batch_size, bno, imgdim)
        X = X_train_orig/255
        _, Y = hv.convert_to_one_hot(classes, Y_train_orig)

        batch_data.put((X,Y))
        #print("loaded batch {}".format(bno))

        bno += 1


def trainOnData(dataPath, saveDest , train_loss, num_epochs, mini_batch_size, imgdim=(100,100,1)):

    print("Making data set from path ", dataPath)

    num_train_files, num_dev_files, tmp_keys, train_files_list, dev_files_list, classes = hv.make_dataset(dataPath, 0.2, imgdim)
    num_mini_batches = num_train_files // mini_batch_size


    print ("number of training examples = " + str(num_train_files))
    print ("number of test examples = " + str(num_dev_files))

    model = makeModel(imgdim, len(classes))
    model.compile(loss=train_loss, optimizer='adam', metrics=['accuracy'])

    for epoch in range(0, num_epochs):

        #shuffle the file list
        shuffle(train_files_list)

        threading.Thread(target = minibatchLoader, args=(classes, train_files_list,num_mini_batches, mini_batch_size, 0, imgdim)).start()

        print("\nEpoch {}/{}".format(epoch,num_epochs))

        for bno in tqdm(range(0,num_mini_batches)):
            #t_1 = time.time()
            X,Y = batch_data.get()
            #t_2 = time.time()
            model.train_on_batch(X, Y)
            #t_3 = time.time()
			
            #print("Took [{}] s to load, and [{}] s to train on mini batch".format((t_2-t_1),(t_3-t_2)))

        testModel(model, num_dev_files, dev_files_list, classes, mini_batch_size, imgdim)

    #done with mini batches, now test
    print("\nRunning final evaluation.....")

    X_test_orig, Y_test_orig = hv.load_minibatch(classes, dev_files_list, num_dev_files, 0, imgdim)
    X_test = X_test_orig/255
    _, Y_test = hv.convert_to_one_hot(classes, Y_test_orig)

    preds = model.evaluate(X_test, Y_test, batch_size=mini_batch_size)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    hv.saveModel(model, saveDest, classes)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data.path", required=True,
        help="folder containing sub folders with images")
    ap.add_argument("-dl", "--dest.loc", required=True,
        help="where to save the trained model")
    ap.add_argument("-iw", "--img.width", required=True,
        help="width of the image")
    ap.add_argument("-ih", "--img.height", required=True,
        help="height of the image.")
    ap.add_argument("-ic", "--img.chan", default=3,
        help="channels in the image")
    ap.add_argument("-te", "--train.epoch", default=2,
        help="number of epochs")
    ap.add_argument("-tl", "--train.loss", default='categorical_crossentropy',
        help="loss function")
    ap.add_argument("-mb", "--minibatch.size", default=1024,
        help="size of images to include in each mini batch, keep to a multiple of 2")

    args = vars(ap.parse_args())

    imgdim = (int(args["img.height"]), int(args["img.width"]), int(args["img.chan"]))
    trainOnData(args["data.path"], args["dest.loc"], args["train.loss"], int(args["train.epoch"]), int(args["minibatch.size"]),imgdim)

    print("\n\nDone...")