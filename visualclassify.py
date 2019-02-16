from PIL import Image
import pprint
import tkinter as tk
from tkinter import filedialog

import sys
import numpy as np
import pprint
import time
import os
import argparse as argparse
import json
import hvutils as hv
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

from keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History
from keras.initializers import glorot_uniform

ap = argparse.ArgumentParser()
ap.add_argument("-mp", "--model.path", required=True,
        help="folder containing the model")
ap.add_argument("-ml", "--model.loss", required=True,
        help="either binary_crossentropy or categorical_crossentropy")
ap.add_argument("-iw", "--img.width", required=True,
        help="width of the image")
ap.add_argument("-ih", "--img.height", required=True,
    help="height of the image.")
ap.add_argument("-ic", "--img.chan", default=3,
        help="channels in the image")

args = vars(ap.parse_args())

imgdim = (int(args["img.height"]), int(args["img.width"]), int(args["img.chan"]))
model, classes, graph, sess = hv.loadModel(args["model.path"], args["model.loss"])

print("Loaded Classes...")
pprint.pprint(classes)

def classifyImage(img):
        
    begin = time.time()
    with graph.as_default():
            with sess.as_default():
                predictions = model.predict(img)
    end = time.time()
    max_score_index = np.argmax(predictions[0])
    print("Took {} s, class and confidence are [{}] [{}] ".format(end-begin, classes[max_score_index], predictions[0][max_score_index]))

    for i in range(0, len(classes)):
        print("[{}] [{}]".format(classes[i], predictions[0][i]))
    
def classifyLoop():

    tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    while True:
        filePath = filedialog.askopenfilename(filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if len(filePath) is 0:
            break

        print("Processing file: ", filePath)

        rootPath = os.path.dirname(filePath)
        filename = os.path.basename(filePath)

        img = hv.load_image_for_classification(filePath, imgdim)
        classifyImage(img)

#extract the plates
classifyLoop()
