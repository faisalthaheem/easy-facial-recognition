import argparse
import sys
import os
import tensorflow as tf
import shutil as sh
from PIL import Image
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-sd", "--source.dir", required=True,
	help="Directory containing files to be resized")	
ap.add_argument("-bd", "--bad.dir", required=True,
	help="Directory containing files that could not be processed")	
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("conv2gs.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def validateImageChannels(filePath):
	try:
		
		img = Image.open(filePath)
		if img.mode != 8:
			img = img.convert("L")
			img.save(filePath)

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
results = Parallel(n_jobs=4, backend="threading")(map(delayed(validateImageChannels), items_to_process))
items_to_process = []
logging.info("Done..")
