'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps

class ObjectDetector():

    def __init__(self):
        self.__LOAD_MODEL = False
        self.model_name = ''

    def train(self, images, labels):

        images = self.preprocess(images)

        if self.__LOAD_MODEL:
            model = self.load_model()
        else:
            model = self.create_model()

    
    def preprocess(self, images):

        '''
    
            Preprocess images by converting them to a matrix of 
            grayscale values, and then converting the values to 
            decimal between 0 and 1

            Params:
                images (dict) : dictionary mapping image filenames 
                                to PIL images

            Returns:
                processed (dict) : dictionary mapping images filenames 
                                   to grayscale matricies

        '''

        processed = {}
        for file_name in images:
            image = images[file_name]
            grayscale_image = ImageOps.grayscale(image)
            image_matrix = np.array(grayscale_image)
            image_matrix = image_matrix / 255
            processed[file_name] = image_matrix
        return processed

    def load_model(self):
        pass

    def create_model(self):
        print("Creating model")

    def test(self, rgb, grayscale, labels):
        pass

    def new(self, original, grayscale):
        return grayscale