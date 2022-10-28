'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps 

class ObstacleDetector():

    def __init__(self):
        self.__LOAD_MODEL = False
        self.model_name = ''

    def train(self, rgb, grayscale, labels):

        if self.__LOAD_MODEL:
            model = self.load_model()
        else:
            model = self.create_model()

        print(grayscale[0].shape)

    def load_model(self):
        pass

    def create_model(self):
        pass

    def test(self, rgb, grayscale, labels):
        pass

    def new(self, original, grayscale):
        return grayscale