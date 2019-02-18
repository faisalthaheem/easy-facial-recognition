import argparse
import sys
import os
import shutil as sh
from PIL import Image
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-sd", "--source.dir", required=True,
    help="Directory containing files to extract faces from")	
ap.add_argument("-bd", "--bad.dir", required=True,
    help="Directory containing files in which no faces could be found")
ap.add_argument("-ih", "--img.height", required=True,
    help="Height of image")
ap.add_argument("-iw", "--img.width", required=True,
    help="Width of image")	
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("extract-faces.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def extractFace(filePath):
    
    try:
        faceCascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
        
        image = cv2.imread(filePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        #logging.info("Found [{}] face(s) in [{}]".format(len(faces),filePath))

        if len(faces) == 1:
            x,y,w,h = faces[0]
            faceImage = image[y:y+h, x:x+w]
            faceImage = cv2.resize(faceImage, (int(args['img.width']),int(args['img.height'])))
            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filePath, faceImage)

            #logging.info("Found [{}] face(s) in [{}]".format(len(faces),filePath))

        else:
            sh.move(filePath, os.path.join(args['bad.dir'], os.path.basename(filePath)))

    except Exception as e:
        logging.error(e)
        logging.info("error processing [{}]".format(filePath))


# verify the directories exist
if not os.path.exists(args["source.dir"]) or not os.path.isdir(args["source.dir"]):
    logging.error("%s is not a directory or does not exist." % args["source.dir"])
    sys.exit()

#list containing dest images to process
items_to_process = []

logging.info("Processing files in " + args["source.dir"])
for root, dirs, files in os.walk(args["source.dir"]):
    
    totalFiles = len(files)
    logging.info("[{}] files to process".format(totalFiles))

    for i in tqdm(range(0,totalFiles)):
        fileName = files[i]        
        filePath = os.path.join(root,fileName)
        
        items_to_process.append(filePath)

logging.info("Commencing parallel processing jobs")			
results = Parallel(n_jobs=4, backend="threading")(map(delayed(extractFace), items_to_process))
items_to_process = []
logging.info("Done..")
