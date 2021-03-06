from preprocessing.meanpreprocessor import MeanPreprocessor
#from preprocessing.patchpreprocessor import PatchPreporcessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from inputOutput.hdf5datasetgenerator import HDF5DataGenerator
from dataset.simpledatasetloader import SimpleDatasetLoader
from preprocessing.imagetoarraypreprocessor import ImageToArrayProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from nn.cnn.vgg import MiniVGGNet
from nn.cnn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np
import argparse
import cv2
from keras.utils import to_categorical
import matplotlib
from keras.callbacks import ModelCheckpoint 
from flask import Flask
from keras.callbacks import LearningRateScheduler 
# app = Flask(__name__)

# if __name__== '__main__':
#     app.run(debug="True")


def step_decay(epohs):

    initAlpha = 0.01
    factor=0.25
    dropEvery=5

    alpha = initAlpha*(factor**np.floor((1+epohs)/dropEvery))
    return float(alpha)

matplotlib.use("Agg")


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
    help="Path to the output loss/accuracy plot")
ap.add_argument("-m", "--model", required=False,
    help="Path to the model saved")
args = vars(ap.parse_args())



# img_path="drive/My Drive/data_sleeves"
# print("going to load images")
# imagePaths = np.array(list(paths.list_images(img_path)))

train_hdf5 = "ShirtSleeves/hdf5data/train.hdf5"
test_hdf5 = "ShirtSleeves/hdf5data/test.hdf5"


sp = SimplePreprocessor(128,128)
iap = ImageToArrayProcessor()


# sdl = SimpleDatasetLoader(preprocessor=[sp,iap])
# (data,label) = sdl.load(imagePaths,verbose=500)
# #print("labels ",label)

# le = LabelEncoder()
# labels = le.fit_transform(label)

# try:
#     data = data.astype("float32")/255.0
# except Exception as e:
#     print("error as ",str(e)) 

aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
                        shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

trainGen = HDF5DataGenerator(train_hdf5,28,aug=aug,preprocessors=[sp,iap],classes=4)
#testGen  = HDF5DataGenerator(config.TEST_HDF5,128,aug=aug,preprocessors=[sp,pp,mp,iap],classes=2)
valGen   = HDF5DataGenerator(test_hdf5,28,aug=aug,preprocessors=[sp,iap],classes=4)

#(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=40)



#trainY = to_categorical(trainY)
#testY = to_categorical(testY)
import os
fname=os.path.sep.join(["ShirtSleeves/model/","weight-{epoch:03d}-{val_loss:.4f}.hdf5"])
callbacks=[ModelCheckpoint(fname,monitor="val_loss",mode="min",save_best_only=True)]
##,LearningRateScheduler(step_decay)

#opt = Adam(lr=0.01,beta_1=0.9, beta_2=0.999, epsilon=None)
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model = MiniVGGNet.build(width=128, height=128, depth=3, classes=4)

model = ResNet.build(128,128,3,4,(9,9,9),(64,64,128,256),reg=0.0005)
# model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
#                     steps_per_epoch=len(trainX) / 32, epochs=epochs)

model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


H= model.fit_generator(trainGen.generator(),steps_per_epoch= trainGen.numImages//28 ,
                    validation_data = valGen.generator(),validation_steps=valGen.numImages//28,
                    verbose=1,epochs=70,callbacks=callbacks)

#model.save(config.MODEL_PATH,overwrite=True)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 70), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 70), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 70), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 70), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAr-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("ShirtSleeves/output/")
