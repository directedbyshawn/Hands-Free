'''

    Driver for hands free image assistance.

'''

from lib.object_detection import ObjectDetector
from sys import argv
from os import listdir, mkdir
from os.path import exists, isfile, isdir
import json
import config as cfg

'''

    RUN TIME ARGUMENTS:
        - 1: Input type
            - 1: Single image (JPG/PNG)
            - 2: Directory of images (JPG/PNG)
            - 3: Video (mp4)
            - 4. Training
        - 2: Path to directory or file (no path for training)

    ex. single image, directory, video, training
        - python run.py 1 path/to/image.jpg
        - python run.py 2 path/to/images
        - python run.py 3 path/to/video.mp4
        - python run.py 4

'''

object_detector = ObjectDetector(
    training_size=cfg.OD_TRAINING_SIZE,
    validation_size=cfg.OD_VALIDATION_SIZE
)

def main():

    global object_detector
    
    # parse arguments
    assert int(argv[1])
    assert int(argv[1]) >= 1 and int(argv[1]) <= cfg.INPUT_TYPES
    input_type = int(argv[1])
    if input_type != 4:
        assert argv[2]
        path = argv[2]
    else:
        path = None

    # validate input data
    if input_type == 1 or input_type == 3:

        # single image or video exists
        path = argv[2]
        assert isfile(path)
        assert exists(path)

        # file type validation
        if (input_type == 1):
            assert path.lower().endswith(cfg.IMAGE_TYPES)
            single_image(path)
        else:
            assert path.lower().endswith(cfg.VIDEO_TYPES)
            video(path)

    elif input_type == 2:

        # path exists and is not empty
        path = argv[2]
        assert isdir(path)
        assert len(listdir(path)) != 0

        # all files are images with valid type
        for file_name in listdir(path):
            assert file_name.lower().endswith(cfg.IMAGE_TYPES)

        directory_images(path)

    elif input_type == 4:

        assert exists('data/images/training')
        assert len(listdir('data/images/training')) != 0
        assert exists(cfg.OD_TRAINING_LABELS_PATH)
        if cfg.OD_VALIDATE:
            assert exists('data/images/validation')
            assert len(listdir('data/images/validation')) != 0
            assert exists(cfg.OD_VALIDATION_LABELS_PATH)

        train()

    else:
        raise InvalidInput

def make_output_dir():
    index = len(listdir('output'))
    output_dir = f'output/{index}'
    mkdir(output_dir)
    return output_dir, index

def single_image(path):

    # output dir is the path to the directory, index
    # is the folder number. ex. output/1, output/2, etc.
    output_dir, index = make_output_dir()

    # run image through object detection model
    image = object_detector.predict(path)

    image.save(f'{output_dir}/image.jpg')

def directory_images(path):
    
    output_dir, index = make_output_dir()

    for index, file_name in enumerate(listdir(path)):
        image = object_detector.predict(f'{path}/{file_name}')
        image.save(f'{output_dir}/image{index}.jpg')

def video(path):
    pass

def train():

    # load labels
    object_detector.training_labels = load_labels(cfg.OD_TRAINING_LABELS_PATH)
    object_detector.validation_labels = load_labels(cfg.OD_VALIDATION_LABELS_PATH)

    if cfg.TRAIN_OBSTACLES:
        object_detector.train()

def load_labels(path):

    ''' Load labels from json file '''

    labels = None
    with open(path) as file:
        labels = json.load(file)
    return labels

if __name__ == '__main__':
    main()

class InvalidInput(Exception):
    def __init__(self):
        self.message = "ERROR: Invalid input"
        super(message=self.message)