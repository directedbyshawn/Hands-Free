'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

from .instance_data import Instance, Object, Box
from xml.etree.ElementTree import Element, SubElement, ElementTree
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import ImageOps
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
import torch
from time import sleep

CLASS_MAP = {
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

class ObjectDetector():

    def __init__(self, training_size):
        self.__SAVE_MODEL = True
        self.__LOAD_MODEL = True
        self.__TRAINING_SIZE = training_size
        self.__data_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = list(CLASS_MAP.keys())
        self.model = ''
        self.originals = {}
        self.preprocessed = {}
        self.labels = []
        self.training_instances = []
        self.dataset = None

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
                if object_raw['category'] not in CLASS_MAP.keys():
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
        
        # generate xml files for each training instance
        self.generate_xml()

        # create dataset from labels and images
        self.dataset = core.Dataset('data/labels/train', 'data/train')

        # train model on dataset
        self.model = core.Model(classes=self.classes, device=self.device)
        self.model.fit(self.dataset, epochs=6, verbose=True)

        if self.__SAVE_MODEL:
            self.model.save('models/faster_rcnn_5.pth')


    def predict(self, path):

        model_path = 'models/faster_rcnn_5.pth'
        self.model = core.Model(classes=self.classes, device=self.device)
        self.model = core.Model.load(model_path, classes=list(CLASS_MAP.keys()))

        image_path = 'data/train/00a2f5b6-d4217a96.jpg'
        image = utils.read_image(image_path)
        predictions = self.model.predict(image)

        labels, boxes, scores = predictions
        print(labels)
        print(boxes)
        print(scores)
        show_labeled_image(image, boxes, labels)

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

    def generate_xml(self):

        '''

            Generate xml files for each training instance

        '''

        for instance in self.training_instances:

            # reuse duplicate documents
            path = f'data/labels/train/{instance.file_name[:-4]}.xml'
            if os.path.exists(path):
                continue

            # create document
            root = Element('annotation')
            SubElement(root, 'folder').text = 'data/train'
            SubElement(root, 'filename').text = instance.file_name
            SubElement(root, 'path').text = f'/data/train/{instance.file_name}'
            source = SubElement(root, 'source')
            SubElement(source, 'database').text = 'BDD 100K'
            size = SubElement(root, 'size')
            SubElement(size, 'width').text = str(instance.original.width)
            SubElement(size, 'height').text = str(instance.original.height)
            SubElement(size, 'depth').text = '3'
            SubElement(root, 'segmented').text = '0'
            for object in instance.objects:
                object_xml = SubElement(root, 'object')
                SubElement(object_xml, 'name').text = object.class_name
                SubElement(object_xml, 'pose').text = 'Unspecified'
                SubElement(object_xml, 'truncated').text = '0'
                SubElement(object_xml, 'difficult').text = '0'
                box = SubElement(object_xml, 'bndbox')
                SubElement(box, 'xmin').text = str(object.box.x1)
                SubElement(box, 'ymin').text = str(object.box.y1)
                SubElement(box, 'xmax').text = str(object.box.x2)
                SubElement(box, 'ymax').text = str(object.box.y2)
            tree = ElementTree(root)
            tree.write(f'data/labels/train/{instance.file_name[:-4]}.xml')
    


