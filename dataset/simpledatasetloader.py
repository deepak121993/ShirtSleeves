    
import cv2
import numpy as np
import os


class SimpleDatasetLoader:
    def __init__(self,preprocessor=None):
        self.preprocessor = preprocessor


        if self.preprocessor is None:
            self.preprocessor=[]
    

    def load(self,imagePaths,verbose=-1):
        data=[]
        labels=[]


        for (i,imagePath) in enumerate(imagePaths):
            #print("image path in sdl ",imagePath)
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessor  is not None:
                for p in self.preprocessor:
                    try:
                        image = p.preprocess(image)
                    except Exception as e:
                        print("Error with image ",str(e))
            


            data.append(image)
            labels.append(label)

            if verbose>0 and i>0 and (i+1) % verbose==0:
                print("INFO processed {}",format(i+1))

        return (np.array(data),np.array(labels))