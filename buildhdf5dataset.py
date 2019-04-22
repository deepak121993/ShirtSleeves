    
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from inputOutput.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os


img_path="drive/My Drive/data_sleeves"
train_hdf5 = "ShirtSleeves/hdf5data/train.hdf5"
test_hdf5 = "ShirtSleeves/hdf5data/test.hdf5"
trainPaths = list(paths.list_images(img_path))
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
print("trainlables ", trainLabels[0])
le = LabelEncoder()

trainLabels = le.fit_transform(trainLabels)
split = train_test_split(trainPaths,trainLabels,test_size=0.20,stratify=trainLabels,\
        random_state=42)
(trainPaths,testPaths,trainLabels,testLabels)=split

print("trainPaths ",trainPaths[0])
# M = open(config.VAL_MAPPING).read().strip().split("\n")
# M = [r.split("\t")[:2] for r in M]

# valPaths = [os.path.join([config.VAL_IMAGES,m[0]]) for m in M]
# valLabels = le.transform([m[1] for m in M])

datasets = [("train",trainPaths,trainLabels,train_hdf5),("test",testPaths,testLabels,test_hdf5)]

for (dtype,paths,labels,outputPath) in datasets:
    print("[INFO] building {}..".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),128,128,3),outputPath)

    widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widget).start()
   
    for (i,(path,label)) in enumerate(zip(paths,labels)):
        try:
            image = cv2.imread(path)
            image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
            #print("image shape ",image.shape)
            writer.add([image],[label])
            pbar.update(i)
        except Exception as e:
            print("error in Image",str(e))

    pbar.finish()
    writer.close()