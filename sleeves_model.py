from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from AccidentDetection.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from AccidentDetection.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from AccidentDetection.preprocessing.meanpreprocessor import MeanPreprocessor
from AccidentDetection.preprocessing.patchpreprocessor import PatchPreporcessor
from AccidentDetection.preprocessing.simpleProcessor import SimplePreprocessor
from AccidentDetection.preprocessing.croppreprocess import CropPreprocessor
from keras.models import load_model
import numpy as np
import progressbar
import json
from flask import Flask

app = Flask(__name__)

@app.route("/predict/",methods=["GET"])
def predictImage(imgPath):
    print("test")
    PredictDamage.predict(imgPath)


class PredictDamage:

    @staticmethod
    def predict(imgPath):
        #load the pretrained model
        print("[INFO] loading thr pretrained model")
        model = load_model("models/train.hdf5")
        print("[INFO] loaded thr pretrained model")
        sp = SimplePreprocessor(224,224)
        iap = ImageToArrayProcessor()

        sdl = SimpleDatasetLoader(preprocessor=[sp,iap])
        (data,label) =sdl.load(imgPath,verbose=500)

        le = LabelEncoder()
        labels = le.fit_transform(label)
        print("[INFo] data ",data)

        data = data.astype("float32")/255.0
        prediction = model.predict(data)
        print("prediction is  ",prediction)
