'''

    Driver for autonomous vehicle image classification.

'''

# TODO: Add parameter for how many images to train on 

from models.obstacles import ObstacleDetector
from sys import argv
from os import listdir, mkdir
from os.path import exists, isfile, isdir
from PIL import Image, ImageOps
import numpy as np
import json


'''

    RUN TIME ARGUMENTS:
        - 1: Input type
            - 1: Single image (JPG/PNG)
            - 2: Directory of images (JPG/PNG)
            - 3: Video (mp4)
            - 4. Training
        - 2: Path to directory or file

    ex. single image, directory, video
        - python run.py 1 path/to/image.jpg
        - python run.py 2 path/to/images
        - python run.py 3 path/to/video.mp4
        - python run.py 4 number_of_images_to_use

'''

INPUT_TYPES = 4

IMAGE_TYPES = ('.jpg', '.png', '.jpeg')
VIDEO_TYPES = ('.mp4')

TRAIN_OBSTACLES = True
TRAIN_LANES = False
TRAIN_SIGNS = False

TEST_OBSTACLES = True
TEST_LANES = False
TEST_SIGNS = False

TRAINING_SIZE = 100

obstacles = ObstacleDetector()

def main():

    global obstacles
    
    # parse arguments
    assert int(argv[1])
    assert int(argv[1]) >= 1 and int(argv[1]) <= INPUT_TYPES
    input_type = int(argv[1])
    path = ''
    if (input_type != 4):
        assert len(argv) == 3
        path = argv[2]
        assert exists(path)

    # validate input data
    if input_type == 1 or input_type == 3:
        assert isfile(path)
        if (input_type == 1):
            assert path.lower().endswith(IMAGE_TYPES)
        else:
            assert path.lower().endswith(VIDEO_TYPES)
    elif input_type == 2:
        assert isdir(path)
        # directory is not empty
        assert len(listdir(path)) != 0
        # all files are images
        for file_name in listdir(path):
            assert file_name.lower().endswith(IMAGE_TYPES)
    elif input_type == 4:
        assert exists('data/train')
        assert len(listdir('data/train')) != 0
        assert exists('data/labels')
        assert len(listdir('data/labels')) != 0
    else:
        raise InvalidInput
    
    # perform action
    if input_type == 1:

        # SINGLE IMAGE

        # open image and convert to grayscale
        rgb_image = Image.open(path)
        grayscale_image = ImageOps.grayscale(rgb_image)
        grayscale_image.show()

        # test image on obstacle detecting network
        result = rgb_image
        if TEST_OBSTACLES:
            result = obstacles.test(original=rgb_image, grayscale=grayscale_image)

        # save image
        index = len(listdir('output'))
        path = f'output/{index}'
        mkdir(path)
        result.save(f'{path}/result.jpg')

    elif input_type == 4:

        # TRAINING

        # get labels
        label_path = 'data/labels/bdd100k_labels_images_train.json'
        with open(label_path) as file:
            labels = json.load(file)

        # get images from labels
        rgb_images = []
        grayscale_images = []
        count = 0
        for label in labels:
            file_name = label['name']
            path = f'data/train/{file_name}'
            if exists(path):
                rgb_image = Image.open(path)
                rgb_images.append(rgb_image)
                grayscale_image = ImageOps.grayscale(rgb_image)
                grayscale_images.append(grayscale_image)
            count += 1
            if count > TRAINING_SIZE:
                break

        converted_grayscale = convert_grayscale(grayscale_images)

        # obstacles
        obstacles.train(rgb=rgb_images, 
                        grayscale=converted_grayscale,
                        labels=labels)

'''

    Convert grayscale images to matrix of
    decimal values ranging from 0 to 1

'''
def convert_grayscale(images):
    converted = []
    for image in images:
        image_matrix = np.array(image)
        image_matrix = image_matrix / 255
        converted.append(np.array(image_matrix))
    return converted


if __name__ == '__main__':
    main()

class InvalidInput(Exception):
    def __init__(self):
        self.message = "ERROR: Invalid input"
        super(message=self.message)