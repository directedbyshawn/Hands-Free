'''

    Driver for hands free image assistance.

'''

from lib.object_detection import ObjectDetector
from sys import argv
from os import listdir, mkdir
from os.path import exists, isfile, isdir
from PIL import Image
import numpy as np
import json
import config as cfg
import cv2

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
    
def export_signs(original_path, predictions, output_dir):

    ''' Export signs to output directory '''

    signs = []
    original = cv2.imread(original_path)
    labels, boxes, scores = predictions
    for index, label in enumerate(labels):
        if label == 'traffic sign' and scores[index] > cfg.OD_PREDICTION_THRESHOLD:
            box = boxes[index]
            box = [int(val) for val in box]
            xmin, ymin, xmax, ymax = box
            sign = original[ymin:ymax, xmin:xmax]
            bordered_sign = add_border(sign)
            downscaled_image = cv2.resize(bordered_sign, (cfg.SIGN_SIZE, cfg.SIGN_SIZE))
            signs.append(downscaled_image)

    return signs

def add_border(sign):

    height, width, _ = sign.shape

    # border values, ensure aspect ration is 1:1
    amount = np.abs(height - width) // 2
    border1 = border2 = 0
    if np.abs(height - width) % 2 == 0:
        border1 = border2 = amount
    else:
        border1 = amount
        border2 = amount + 1

    # add to top and bottom or left and right
    if height > width: 
        border = [0, 0, border1, border2]
    else:
        border = [border1, border2, 0, 0]

    bordered_sign = cv2.copyMakeBorder(
        sign, 
        border[0], 
        border[1], 
        border[2], 
        border[3], 
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return bordered_sign

def single_image(path):

    # output dir is the path to the directory, index
    # is the folder number. ex. output/1, output/2, etc.
    output_dir, index = make_output_dir()

    # run image through object detection model.
    # predictions is a 3 tuple consisting of a list of labels, a
    # list of bounding box coordinates, and a list of confidence
    image, predictions = object_detector.predict(path)

    # export signs from image, write them to their own directory
    signs = export_signs(path, predictions, output_dir)
    if cfg.SAVE_SIGNS:
        mkdir(f'{output_dir}/signs')
        for i, sign in enumerate(signs):
            cv2.imwrite(f'{output_dir}/signs/sign{i}.jpg', sign)

    # save final image
    image.save(f'{output_dir}/image.jpg')

def directory_images(path):
    
    # output dir is the path to the directory, index
    # is the folder number. ex. output/1, output/2, etc.
    output_dir, index = make_output_dir()

    # run each image in directory through model, export signs from 
    # each image and save them to the signs directory
    for index, file_name in enumerate(listdir(path)):

        # original image
        original_path = f'{path}/{file_name}'

        # run image through object detection model
        image, predictions = object_detector.predict(f'{path}/{file_name}')

        # export signs from image, write them to their own directory
        signs = export_signs(original_path, predictions, output_dir)
        if cfg.SAVE_SIGNS:
            mkdir(f'{output_dir}/signs')
            for i, sign in enumerate(signs):
                cv2.imwrite(f'{output_dir}/signs/image{index}-sign{i}.jpg', sign)

        # save final image
        image.save(f'{output_dir}/image{index}.jpg')

def video(path):
    
    original = cv2.VideoCapture(path)

    output_dir, index = make_output_dir()

    out = cv2.VideoWriter(f'{output_dir}/video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    frames = []
    cont = True
    while cont:
        ret, frame = original.read()
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB)
            frames.append(frame)
        else:
            cont = False
    
    frames_dir = f'{output_dir}/frames'
    mkdir(f'{output_dir}/frames')
    for index, frame in enumerate(frames):
        cv2.imwrite(f'{output_dir}/frames/frame{index}.jpg', frame)

    mkdir(f'{output_dir}/final_frames')
    for index, file_name in enumerate(listdir(frames_dir)):
        image, predictions = object_detector.predict(f'{frames_dir}/{file_name}')
        image_array = np.asarray(image)
        cv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{output_dir}/final_frames/frame{index}.jpg', cv_image)
        out.write(cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    original.release()
    cv2.destroyAllWindows()


def train():

    if cfg.TRAIN_OBSTACLES:
        object_detector.training_labels = load_labels(cfg.OD_TRAINING_LABELS_PATH)
        object_detector.validation_labels = load_labels(cfg.OD_VALIDATION_LABELS_PATH)
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