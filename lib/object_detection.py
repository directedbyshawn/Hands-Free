'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .instance_data import Instance, Object, Box
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import ImageOps

class ObjectDetector():

    def __init__(self, training_size):
        self.__LOAD_MODEL = False
        self.__TRAINING_SIZE = training_size
        self.__data_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.originals = {}
        self.preprocessed = {}
        self.labels = []
        self.training_instances = []
        self.CLASS_MAP = {
            'pedestrian': 1,
            'rider': 2,
            'car': 3,
            'truck': 4,
            'bus': 5,
            'train': 6,
            'motorcycle': 7,
            'bicycle': 8,
            'traffic light': 9,
            'traffic sign': 10
        }

        # use cuda cores if available
        self.check_for_gpu()

    def load_training_data(self, images, labels):

        self.labels = labels

        # keep original copy of images for final manipulation
        self.originals = images

        # preprocess images
        self.preprocessed = self.preprocess(images)

        # create instance class for each training instance
        for index, label in enumerate(labels):
            
            # file name, original rgb, and grayscale matrix
            instance = Instance()
            instance.file_name = label['name']
            instance.original = images[f'data/train/{label["name"]}']
            instance.preprocessed = self.preprocessed[f'data/train/{label["name"]}']

            # objects in image
            objects = []
            for object_raw in label['labels']:
                
                # exclude segmentation labels
                if object_raw['category'] not in self.CLASS_MAP.keys():
                    continue

                # object class
                object = Object()
                object.class_name = object_raw['category']

                # bounding box
                box = Box()
                box.x1 = object_raw['box2d']['x1']
                box.y1 = object_raw['box2d']['y1']
                box.x2 = object_raw['box2d']['x2']
                box.y2 = object_raw['box2d']['y2']
                object.box = box

                objects.append(object)

            instance.objects = objects

            self.training_instances.append(instance)

            if index >= self.__TRAINING_SIZE-1:
                break

        self.__data_loaded = True

    def train(self):
        
        if not self.__data_loaded:
            raise Exception
        
        self.print_instance(0)
    
    def print_instance(self, index):

        '''
        
            Formatted print of training instance
            at specified index

            Params:
                index (int) : index of instance

        '''

        for _ in range(50):
            print('-', end='')
        print()

        instance = self.training_instances[index]
        print(f'Training instance #{index}')
        print(f'File name: {instance.file_name}')
        instance.original.show()
        print(instance.preprocessed)
        print("Objects in image: ")
        for object in instance.objects:
            print(f'\n{object.class_name} with box:')
            print(f'x1: {object.box.x1}, y1: {object.box.y1}\nx2: {object.box.x2}, y2: {object.box.y2}\n')

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

    


