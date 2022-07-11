# Import libraries
import time
import json
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from imutils.video import FPS
import glob

# Import Realsense camera
from realsense_camera import *


# DeepSort Imports
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections
from collections import deque


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of images")
ap.add_argument('--min_score', type=float, default=0.3, help="displays the lowest tracking score.")
ap.add_argument('--input_size', type=float, default=1024, help="input pic size.")
ap.add_argument('--model_feature', type=float, default="model_data/market1501.pb", help="target tracking model file.")
args = vars(ap.parse_args())

# Generate random colors(80 colors equals to detect classes of the model)
colors = np.random.randint(0, 255, (80, 3))


# Derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["dnn"], "frozen_inference_graph_coco.pb"]) 
configPath = os.path.sep.join([args["dnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"]) 

# Loading Mask R-CNN trained on the COCO dataset (80 Classes)
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# Check if we are going to use the GPU
if args["use_gpu"]:
    # set CUDA as the preferable backend and target 
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing the video stream...")

# Load the Realsense Camera 
rs = RealsenseCamera()

fps = FPS.start()

