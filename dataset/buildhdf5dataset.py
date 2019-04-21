    
from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyImageSearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os


img_path="drive/My Drive/data_sleeves"
train_hdf5 = "hdf5data/train.hdf5"
test_hdf5 = "hdf5data/test.hdf5"
trainPaths = list(paths.list_images(img_path))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()

trainLabels = le.fit_transform(trainLabels)
split = train_test_split(trainPaths,trainLabels,test_size=0.20,stratify=trainLabels,\
        random_state=42)
(trainPaths,testPaths,trainLabels,testLabels)=split

# M = open(config.VAL_MAPPING).read().strip().split("\n")
# M = [r.split("\t")[:2] for r in M]

# valPaths = [os.path.join([config.VAL_IMAGES,m[0]]) for m in M]
# valLabels = le.transform([m[1] for m in M])

datasets = [("train",trainPaths,trainLabels,train_hdf5),("test",testPaths,testLabels,test_hdf5)]

for (dtype,paths,labels,outputPath) in datasets:
    print("[INFO] building {}..".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),64,64,3),outputPath)

    widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widget).start()


    pbar.finish()
    writer.close()