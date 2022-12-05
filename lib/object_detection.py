'''

    Model to detect obstacles on the road, such as 
    pedestrians, vehicles, and other 

'''

from .instance_data import Instance, Object, Box, Type
from xml.etree.ElementTree import Element, SubElement, ElementTree
import matplotlib.pyplot as plt
import os
from PIL import ImageOps, Image, ImageFont, ImageDraw, ImageEnhance
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from shutil import rmtree, copyfile, copy
import torch
import config as cfg
from time import sleep
from torchvision import transforms

class ObjectDetector():

    def __init__(self, training_size, validation_size):
        self.__SAVE_MODEL = True
        self.__TRAINING_SIZE = training_size
        self.__VALIDATION_SIZE = validation_size
        self.__data_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__CLASSES = list(cfg.OD_CLASS_MAP.keys())
        self.training_labels = []
        self.validation_labels =[]
        self.model = ''
        self.originals = {}
        self.training_instances = []
        self.validation_instances = []
        self.training_set = None
        self.validation_set = None

    def load_data(self, type):

        # create instance class for each training instance
        labels = self.training_labels if type == Type.TRAINING else self.validation_labels
        max = self.__TRAINING_SIZE if type == Type.TRAINING else self.__VALIDATION_SIZE
        for index, label in enumerate(labels):
            
            # file name, original rgb, and grayscale matrix
            instance = Instance()
            instance.file_name = label['name']

            instance.type = Type.TRAINING if type == Type.TRAINING else Type.VALIDATION

            # objects in image
            objects = []
            for object_raw in label['labels']:
                
                # exclude segmentation labels
                if object_raw['category'] not in self.__CLASSES:
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
            if type == Type.TRAINING:
                self.training_instances.append(instance)
            else:
                self.validation_instances.append(instance)

            if index >= max-1:
                break

        # create xml files
        self.generate_xml(Type.TRAINING if type == Type.TRAINING else Type.VALIDATION)

        self.__data_loaded = True

    def train(self):

        self.load_data(type=Type.TRAINING)
        self.load_data(type=Type.VALIDATION)
        
        if not self.__data_loaded:
            raise Exception
        
        # create training dataset from labels and images
        self.training_set = core.Dataset(
            label_data='data/labels/training', 
            image_folder='data/images/training'
        )


        if cfg.OD_VALIDATE:
            self.validation_set = core.Dataset(
                label_data='data/labels/validation',
                image_folder='data/images/validation'
            )
        else:
            self.validation_set = None
        
        # train model on dataset
        self.model = core.Model(classes=self.__CLASSES, device=self.device)
        losses = self.model.fit(
            self.training_set, 
            val_dataset=self.validation_set,
            epochs=cfg.OD_HYPER['epochs'],
            learning_rate=cfg.OD_HYPER['learning_rate'],
            verbose=True
        )

        if self.__SAVE_MODEL:
            existing = len(os.listdir('models'))
            self.model.save(f'models/faster_rcnn_{existing}.pth')


    def predict(self, image):

        self.model = core.Model(classes=self.__CLASSES, device=self.device)
        self.model = core.Model.load(cfg.OD_MODEL_PATH, classes=self.__CLASSES)

        return self.model.predict(image)

    def annotate_image(self, cv_image, predictions, color):

        labels, boxes, scores = predictions

        image = Image.fromarray(cv_image)
        draw = ImageDraw.Draw(image)

        sign_color = 'blue' if color == 'red' else 'red'

        for index, score in enumerate(scores):
            if score < cfg.OD_PREDICTION_THRESHOLD:
                continue
            box = [int(boxes[index][i]) for i in range(len(boxes[index]))]
            label = labels[index]
            use = sign_color if label not in self.__CLASSES else color
            draw.rectangle(box, outline=use)
            draw.text((box[0], box[1]-12), f'{label} {scores[index]:.2f}', fill=color)

        return image

    def generate_xml(self, instance_type):

        '''

            Generate Pascal VOC XML files for each instance

        '''
        labels = images = ''
        if instance_type == Type.TRAINING:
             labels = 'data/labels/training'
             images = 'data/images/training'
        else:
            labels = 'data/labels/validation'
            images = 'data/images/validation'

        # remove existing files
        rmtree(labels, ignore_errors=True)
        os.mkdir(labels)

        if instance_type == Type.TRAINING:
            instances = self.training_instances
        else:
            instances = self.validation_instances

        for instance in instances:
            if not os.path.exists(instance.get_image_path()):
                continue

            # reuse duplicate documents
            path = f'{labels}/{instance.file_name[:-4]}.xml'
            if os.path.exists(path):
                continue

            # create document
            root = Element('annotation')
            SubElement(root, 'folder').text = images
            SubElement(root, 'filename').text = instance.file_name
            SubElement(root, 'path').text = f'{images}/{instance.file_name}'
            source = SubElement(root, 'source')
            SubElement(source, 'database').text = 'BDD 100K'
            size = SubElement(root, 'size')
            SubElement(size, 'width').text = '1280'
            SubElement(size, 'height').text = '720'
            SubElement(size, 'depth').text = '3'
            SubElement(root, 'segmented').text = '0'
            for object in instance.objects:
                object_xml = SubElement(root, 'object')
                SubElement(object_xml, 'name').text = object.class_name
                SubElement(object_xml, 'pose').text = 'Unspecified'
                SubElement(object_xml, 'truncated').text = '0'
                SubElement(object_xml, 'difficult').text = '0'
                box = SubElement(object_xml, 'bndbox')
                SubElement(box, 'xmin').text = str(int(object.box.x1))
                SubElement(box, 'ymin').text = str(int(object.box.y1))
                SubElement(box, 'xmax').text = str(int(object.box.x2))
                SubElement(box, 'ymax').text = str(int(object.box.y2))
            tree = ElementTree(root)
            tree.write(f'{labels}/{instance.file_name[:-4]}.xml')