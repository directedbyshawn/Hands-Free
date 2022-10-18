'''

    Driver for autonomous vehicle image classification.

'''

from re import A
from turtle import shape
from models.obstacles import ObstacleDetector
from sys import argv
from os import listdir, mkdir
from os.path import exists, isfile, isdir
from PIL import Image, ImageOps
from numpy import asarray, full

'''

    RUN TIME ARGUMENTS:
        - 1: Input type
            - 1: Single image (JPG/PNG)
            - 2: Directory of images (JPG/PNG)
            - 3: Video (mp4)
            - 4. Train
        - 2: Path to directory or file

    ex. single image, directory, video
        - python run.py 1 path/to/image.jpg
        - python run.py 2 path/to/images
        - python run.py 3 path/to/video.mp4

'''

IMAGE_TYPES = ('.jpg', '.png', '.jpeg')
VIDEO_TYPES = ('.mp4')

obstacles = ObstacleDetector()

def main():

    global obstacles
    
    # parse arguments
    assert len(argv) == 3
    assert int(argv[1])
    assert int(argv[1]) >= 1 and int(argv[1]) <= 3
    input_type = int(argv[1])

    # validate input data
    path = argv[2]
    assert exists(path)
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
    else:
        raise InvalidInput

    # single image
    if input_type == 1:

        # open image and convert to grayscale
        rgb_image = Image.open(path)
        grayscale_image = ImageOps.grayscale(rgb_image)
        grayscale_image.show()

        # run image on obstacle detecting network
        result = obstacles.test(original=rgb_image, grayscale=grayscale_image)

        # save image
        index = len(listdir('output'))
        path = f'output/{index}'
        mkdir(path)
        result.save(f'{path}/result.jpg')








if __name__ == '__main__':
    main()

class InvalidInput(Exception):
    def __init__(self):
        self.message = "ERROR: Invalid input"
        super(message=self.message)