'''

    Config for Hands Free

'''

INPUT_TYPES = 4

IMAGE_TYPES = ('.jpg')
VIDEO_TYPES = ('.mp4', '.mov')

TRAIN_OBSTACLES = True
TRAIN_LANES = False
TRAIN_SIGNS = False

DETECT_OBJECTS = True
CLASSIFY_SIGNS = False
DETECT_LANES = False

OD_TRAINING_SIZE = 30000
OD_VALIDATION_SIZE = 3000

OD_VALIDATE = True

OD_TRAINING_LABELS_PATH = 'data/labels/bdd100k_labels_images_train.json'
OD_VALIDATION_LABELS_PATH = 'data/labels/bdd100k_labels_images_val.json'

OD_CLASS_MAP = {
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

OD_HYPER = {
    'epochs': 10,
    'batch_size': 2,
    'learning_rate': 0.002
}

SIGN_SIZE = 32
SAVE_SIGNS = True

OD_PREDICTION_THRESHOLD = 0.85

OD_MODEL_PATH = 'models/faster_rcnn_5000_instances_12_epochs.pth'

# traffic sign classification constants
TSC_PREDICTION_THRESHOLD = 0.85
# TSC_PREDICTION_THRESHOLD = 0.90
# TSC_MODEL_NAME = 'traffic_sign_recognition_eight-classes.h5' # greyscaled images
TSC_MODEL_NAME = 'tsc_rgb_eightClasses.h5'