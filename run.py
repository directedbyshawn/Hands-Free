'''

    Driver for hands free image assistance.

'''
from lib.object_detection import ObjectDetector
from sys import argv
from os import listdir, mkdir, system, name
from os.path import exists, isfile, isdir
import numpy as np
import json
import config as cfg
import cv2
import progressbar
from time import sleep

import matplotlib.pyplot as plt

from lib.sign_classification import classify_frame, load_TSC_model
from lib.lane_detection import detect_lanes
import torch

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

def export_signs_from_frame(frame, predictions):

    ''' Export signs to output directory '''

    signs = []
    labels, boxes, scores = predictions
    for index, label in enumerate(labels):
        if label == 'traffic sign' and scores[index] > cfg.OD_PREDICTION_THRESHOLD:
            box = boxes[index]
            box = [int(val) for val in box]
            xmin, ymin, xmax, ymax = box
            sign = frame[ymin:ymax, xmin:xmax]
            bordered_sign = add_border(sign)
            downscaled_image = cv2.resize(bordered_sign, (cfg.SIGN_SIZE, cfg.SIGN_SIZE))
            signs.append(downscaled_image)

    return signs


def predict_traffic_signs(image, predictions):
    # Predict traffic signs and return a new tuple of predictions

    tsc_model = load_TSC_model(cfg.TSC_MODEL_NAME)
    labels, boxes, scores = predictions
    scores_list = scores.tolist()

    for index, label in enumerate(labels):

        if label == 'traffic sign' and scores[index] > cfg.OD_PREDICTION_THRESHOLD:
            box = boxes[index]
            box = [int(val) for val in box]
            xmin, ymin, xmax, ymax = box

            sign = image[ymin:ymax, xmin:xmax]
            bordered_sign = add_border(sign)
            downscaled_image = cv2.resize(bordered_sign, (cfg.SIGN_SIZE, cfg.SIGN_SIZE))

            sign_class, sign_name, sign_accuracy = classify_frame(tsc_model, downscaled_image)

            if sign_accuracy > cfg.TSC_PREDICTION_THRESHOLD:
                if sign_name == 'other':
                    # predict 'other' class as 'traffic sign'
                    continue

                labels[index] = sign_name
                scores_list[index] = sign_accuracy
             

    predictions = (labels, boxes, torch.Tensor(scores_list))

    return predictions

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

    system('cls') if name == 'nt' else system('clear')
    print(f'\n\nSingle image: {path}\n\n')

    # output dir is the path to the directory, index
    # is the folder number. ex. output/1, output/2, etc.
    output_dir, index = make_output_dir() # uncomment it

    # run image through object detection model.
    # predictions is a 3 tuple consisting of a list of labels, a
    # list of bounding box coordinates, and a list of confidence
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    predictions = object_detector.predict(image)
    
    # detect lanes at this point
    if cfg.DETECT_LANES:
        image = detect_lanes([image])
        
    # predict traffic signs and write it to predictions
    if cfg.CLASSIFY_SIGNS:
        predictions = predict_traffic_signs(image, predictions)

    # signs = export_signs(path, predictions, output_dir)
    # if cfg.SAVE_SIGNS:
    #     mkdir(f'{output_dir}/signs')
    #     for i, sign in enumerate(signs):
    #         cv2.imwrite(f'{output_dir}/signs/sign{i}.jpg', sign)

    image = object_detector.annotate_image(image, predictions, color='red')
       
    # save final image
    image.save(f'{output_dir}/image.jpg')

def directory_images(path):

    system('cls') if name == 'nt' else system('clear')
    print(f'\n\nDirectory: {path}\n\n')
    
    # output dir is the path to the directory, index
    # is the folder number. ex. output/1, output/2, etc.
    output_dir, index = make_output_dir()

    # progress bar
    bar = progressbar.ProgressBar(maxval=100, \
        widgets=[progressbar.Bar('=', 'progress: [', ']'), '', progressbar.Percentage()])
    increment = 100 / len(listdir(path))
    progress = 0
    bar.start()

    # run each image in directory through model, export signs from 
    # each image and save them to the signs directory
    for index, file_name in enumerate(listdir(path)):

        # original image
        original_path = f'{path}/{file_name}'
        original = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)

        # run image through object detection model
        predictions = object_detector.predict(original)
        
        # predict traffic signs and write it to predictions
        if cfg.CLASSIFY_SIGNS:
            predictions = predict_traffic_signs(original, predictions)
            
        # export signs from image, write them to their own directory
        # signs = export_signs(original_path, predictions, output_dir)

        # change name in prediction list. It is tuple ((predictions, bb, scores))

        # if cfg.SAVE_SIGNS:
        #     mkdir(f'{output_dir}/signs') if not exists(f'{output_dir}/signs') else None
        #     for i, sign in enumerate(signs):
        #         cv2.imwrite(f'{output_dir}/signs/image{index}-sign{i}.jpg', sign)

        # save final image
        image = object_detector.annotate_image(original, predictions, color='red')
        
        # detect lanes at this point
        if cfg.DETECT_LANES:
            image = detect_lanes([image])
            
        image.save(f'{output_dir}/image{index}.jpg')

        progress += increment
        if progress > 100:
            progress = 100
        bar.update(progress)

    bar.finish()

def to_frames(tuple):

    return (tuple[0]*60+tuple[1])*24

def video(path):

    system('cls') if name == 'nt' else system('clear')
    print(f'\n\nVideo: {path}\n\n')
    
    original = cv2.VideoCapture(path)

    output_dir, index = make_output_dir()

    # output video buffer
    out = cv2.VideoWriter(f'{output_dir}/video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # save each frame from video to array
    frames = []
    cont = True
    clip = False
    start_time = (3, 30)
    end_time = (3, 45)
    assert to_frames(start_time) < to_frames(end_time)
    count = 0

    while cont:
        if clip:
            ret, frame = original.read()
            if ret:
                if count >= to_frames(start_time) and count <= to_frames(end_time):
                    frames.append(frame)
                if count > to_frames(end_time):
                    cont = False
                count += 1
            else:
                cont = False
        else:
            ret, frame = original.read()
            if ret:
                frames.append(frame)
            else:
                cont = False
    
    # progress bar
    bar = progressbar.ProgressBar(maxval=100, \
        widgets=[progressbar.Bar('=', 'progress: [', ']'), '', progressbar.Percentage()])
    increment = 100 / len(frames)
    progress = 0
    bar.start()

    # run each frame through model, write frame with object detection to buffer
    for index, frame in enumerate(frames):
        
        predictions = object_detector.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cfg.CLASSIFY_SIGNS:
            predictions = predict_traffic_signs(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), predictions)

        image = object_detector.annotate_image(frame, predictions, color='blue')
        
        # detect the lanes
        if cfg.DETECT_LANES:
            image = detect_lanes([image])
            
        # convert to RGBcv & write to buffer
        image_array = np.asarray(image)
        out_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        out.write(out_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        progress += increment 
        bar.update(progress if progress < 100 else 100 )

    bar.finish()

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
