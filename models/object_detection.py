'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

class ObjectDetector():

    def __init__(self):
        self.__LOAD_MODEL = False
        self.model_name = ''

    def train(self, images, labels):

        print("TRAINING")

        images = self.preprocess(images)

        print(images[0])

        if self.__LOAD_MODEL:
            model = self.load_model()
        else:
            model = self.create_model()
   
    def preprocess(self, images):
        processed = []
        for image in images:
            grayscale_image = ImageOps.grayscale(image)
            image_matrix = np.array(grayscale_image)
            image_matrix = image_matrix / 255
            processed.append(np.array(image_matrix))
        return processed

    def load_model(self):
        pass

    def create_model(self):
        pass

    def test(self, rgb, grayscale, labels):
        pass

    def new(self, original, grayscale):
        return grayscale