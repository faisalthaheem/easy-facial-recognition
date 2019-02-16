import threading
import time
import base64
import sys, os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import json
import signal
import traceback
import cv2
import numpy as np
from pprint import pprint
import cv2
import hvutils as hv
from PIL import Image

from PIL import Image
import tkinter as tk
from tkinter import filedialog

import argparse as argparse
from skimage.transform import resize
from skimage.color import rgb2gray

from keras import layers
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import History
from keras.initializers import glorot_uniform

try:
    from gouge.colourcli import Simple
    Simple.basicConfig(level=0)
except ImportError:
    import logging
    logging.basicConfig(level=0)

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
pprint(classes)


os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "./dots"
os.putenv('GST_DEBUG_DUMP_DIR_DIR', './dots')
os.putenv('GST_DEBUG', '0')

LOG = logging.getLogger(__name__)

queued_frames = []

frame_grabber = None

GObject.threads_init()
Gst.init(None)

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


class MainPipeline():
    def __init__(self):
        self.pipeline = None
        self.videosrc = None
        self.videoparse = None
        self.videosink = None
        self.current_buffer = None

    def on_error(self, bus, msg):
        print('Error {}: {}, {}'.format(msg.src.name, *msg.parse_error()))

    def bus_call(self, bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            LOG.debug('End of Stream')
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            LOG.error('!! Error %s: Debug INFO: %s', err, debug)
            loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            LOG.debug('State changed from %s to %s (pending=%s)',
                    old.value_name, new.value_name, pending.value_name)
        elif message.type == Gst.MessageType.STREAM_STATUS:
            type_, owner = message.parse_stream_status()
            LOG.debug('Stream status changed to %s (owner=%s)',
                    type_.value_name, owner.name)
        elif message.type == Gst.MessageType.DURATION_CHANGED:
            LOG.debug('Duration changed')
        else:
            LOG.debug('!! Unknown message type: %r', message.type)
        return True

    def pad_factory(self, element):
        def pad_added(src, pad):
            target = element.sinkpads[0]
            pad_name = '%s:%s' % (pad.get_parent_element().name, pad.name)
            tgt_name = '%s:%s' % (target.get_parent_element().name, target.name)
            LOG.debug('New dynamic pad %s detected on %s. Auto-linking it to %s',
                    pad_name, src.name, tgt_name)
            pad.link(target)
        return pad_added

    def pull_frame(self, sink):
        # second param appears to be the sink itself
        # good sample - https://hackaday.io/project/14729-iss-hdev-image-availability/log/47520-python-script
        #print("in pull frame")
        sample = sink.emit("pull-sample")
        if sample is not None:

            # caps = sample.get_caps()
            # height = caps.get_structure(0).get_value('height')    
            # width = caps.get_structure(0).get_value('width')

            # print(height,width)

            current_buffer = sample.get_buffer()
            current_data = current_buffer.extract_dup(0, current_buffer.get_size())
         
            #send_all(current_data)
            #queued_frames.append(current_data)
            arr,img = hv.bytes_to_image_for_classification(current_data, "L")
            sample = None
            current_data = None
            current_buffer = None

            queued_frames.append((arr,img))


        return Gst.FlowReturn.OK

    def gst_thread(self):
        print("Initializing GST Elements")

        #lg
        CLI='rtspsrc location=rtsp://192.168.230.106:5554/camera latency=0 ! rtph264depay ! avdec_h264 ! videorate max-rate=15 ! decodebin ! queue max-size-buffers=10 ! jpegenc quality=25 ! jpegparse ! appsink name=sink'

        #bbb
        #CLI='rtspsrc location=rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov latency=0 ! rtph264depay ! avdec_h264 ! videorate max-rate=15 ! decodebin ! queue max-size-buffers=10 ! jpegenc quality=25 ! jpegparse ! appsink name=sink'
        self.pipeline=Gst.parse_launch(CLI)

        # start the video
        print("Setting Pipeline State")
        appsink=self.pipeline.get_by_name("sink")
        appsink.set_property("max-buffers",1)
        appsink.set_property('emit-signals',True)
        appsink.set_property('sync',False) 
        appsink.set_property('wait-on-eos',False)
        appsink.set_property('drop',True)


        appsink.connect('new-sample', self.pull_frame)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        bus = self.pipeline.get_bus()

        image_arr = []
        print("loopin")
        # Parse message
        while True:
            if threading.main_thread().is_alive() is False:
                print("Main thread exited, gst thread exiting")
                return   
                
            message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
            # print "image_arr: ", image_arr
            if len(queued_frames) > 0:
                pass
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(("Error received from element %s: %s" % (
                        message.src.get_name(), err)))
                    print(("Debugging information: %s" % debug))
                    break
                elif message.type == Gst.MessageType.EOS:
                    print("End-Of-Stream reached.")
                    break
                elif message.type == Gst.MessageType.STATE_CHANGED:
                    if isinstance(message.src, Gst.Pipeline):
                        old_state, new_state, pending_state = message.parse_state_changed()
                        print(("Pipeline state changed from %s to %s." %
                            (old_state.value_nick, new_state.value_nick)))
                else:
                    pprint(message)
                    print("Unexpected message received.")

        # Free resources
        self.pipeline.set_state(Gst.State.NULL)

def showFrames():

    while True:
        if threading.main_thread().is_alive() is False:
            print("Main thread exited, showFrames exiting")
            return        

        if len(queued_frames) > 0:
            print("q size[{}]".format(len(queued_frames)))
            #queued_frames.pop()
            arr,img = queued_frames.pop()
            #print("{}".format(img.size))
            x = image.img_to_array(img.resize((128,128))).astype(np.uint8)
            x = np.expand_dims(x,axis=0)
            x = x / 255

            classifyImage(x)

            cv2.imshow("preview", arr)
            cv2.waitKey(1)
        else:
            time.sleep(0)
            

if __name__ == "__main__":

    pipeline = MainPipeline()
    gst_thread = threading.Thread(target=pipeline.gst_thread)
    gst_thread.start()

    frame_thread = threading.Thread(target=showFrames)
    frame_thread.start()
    frame_thread.join()
    
    
    print("exiting")