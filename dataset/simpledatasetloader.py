    
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
            try:
                if self.preprocessor  is not None:
                    for p in self.preprocessor:
                        image = p.preprocess(image)
                        
                data.append(image)
                labels.append(label)
            
            except:
                print("error with image")
                continue


            if verbose>0 and i>0 and (i+1) % verbose==0:
                print("INFO processed {}",format(i+1))

        print("labels ",labels)
        return (np.array(data),np.array(labels))