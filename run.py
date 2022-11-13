'''

    Driver for autonomous vehicle image assistance.

'''

from lib.object_detection import ObjectDetector
from sys import argv
from os.path import exists
import json
import config

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
        - python run.py 4

'''

object_detector = ObjectDetector(training_size=config.OD_TRAINING_SIZE)

def main():

    global object_detector
    
    # parse arguments
    #assert int(argv[1])
    #assert int(argv[1]) >= 1 and int(argv[1]) <= INPUT_TYPES
    input_type = int(argv[1])
    path = ''

    # validate input data
    if input_type == 1 or input_type == 3:
        path = argv[2]
        '''
        assert isfile(path)
        assert len(argv) == 3
        assert exists(path)
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

    '''
    
    # perform action
    if input_type == 1:

        # SINGLE IMAGE

        path="test.jpg"

        object_detector.predict(path)

        # open image and convert to grayscale

        
        '''
        rgb_image = Image.open(path)
        grayscale_image = ImageOps.grayscale(rgb_image)
        grayscale_image.show()

        # test image on obstacle detecting network
        result = rgb_image
        if TEST_OBSTACLES:
            result = object_detector.test(original=rgb_image, grayscale=grayscale_image)

        # save image
        index = len(listdir('output'))
        path = f'output/{index}'
        mkdir(path)
        result.save(f'{path}/result.jpg')
        '''

    elif input_type == 4:

        # load labels
        od_labels = load_labels(config.OD_TRAINING_LABELS_PATH)

        if config.TRAIN_OBSTACLES:
            object_detector.train(od_labels)

def load_labels(path):

    ''' Load labels from file '''

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