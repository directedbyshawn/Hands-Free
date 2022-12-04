import numpy as np
import matplotlib.pyplot as plt
import os

import imageio
import glob
import cv2

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model


signs_dict = {"classes": ["stop", "yield", "speed limit", "pedestrian crossing", "do not enter", "other", "school zone", "signal ahead"], "name_to_label": {"stop": 0, "yield": 1, "speed limit": 2, "pedestrian crossing": 3, "do not enter": 4, "other": 5, "school zone": 6, "signal ahead": 7}}
CLASSES_NAMES = signs_dict["classes"]


TRAINING_MODE = False
TESTING_MODE = True
TEST_ON_DIFFERENT_DATASET = False
SAVE_MODEL = False

SHOW_IMAGE_AND_PREDICTION_IN_TESTING = False

MODEL_NAME = "secondModel_eightClasses.h5"
NUM_CLASSES = 8

def divide_data(train, test):
    # set aside 20% of train and test data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        train, test, test_size=0.2, shuffle=True, random_state=42)

    # Use the same function above for the validation set
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, shuffle=True, random_state=42)  # 0.25 x 0.8 = 0.2

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_predict():
    images = []

    # path = os.getcwd() + "/data/predict/"
    # path = os.getcwd() + "/data/predict/VT/"
    # extension = "*.png"


    path = os.getcwd() + "/data/predict/signs/"
    # path = os.getcwd() + "/data/predict/signs_1/"
    extension = "*.jpg"



    for im_path in glob.glob(path + extension):
        image = imageio.v2.imread(im_path)
        # print(image.shape)
        images.append(image)

    # print(len(images))
    return images


def train(X_train, X_valid, y_train, y_valid):
    """ Return CNN model """
    print("Start training")


    print("The model is building")
    model = build_model()
    print(model.summary())

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=10)
    print(history)

    print("Training is ended")

    if SAVE_MODEL:
        model.save("models/" + MODEL_NAME)

    return model

def load_training_data():
    # count = 0

    train = []
    test = []
    for n_class in range(0, NUM_CLASSES):

        path = os.getcwd() + "/data/" + str(n_class) + "/"

        class_train = []
        # print(path)
        for im_path in glob.glob(path + "*.png"):
            image = imageio.v2.imread(im_path)

            class_train.append(image)

        # to "normalize" the number of each class
        # if len(class_train) > 150:
        #     np.random.shuffle(class_train)
        #     class_train = class_train[:150][:][:]

        train.extend(class_train)
        test.extend([n_class] * len(class_train))

    # print(train[0])
    # print(test[0])
    train = np.array(train)
    test = np.array(test)


    X_train, X_valid, X_test, y_train, y_valid, y_test = divide_data(train, test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def test_on_different_dataset():
    # REDO?
    predict_images = load_predict()
    for orig_img in predict_images:
        img = preprocessing(orig_img)
        image = img.reshape(1, 32, 32, 1)

        predict_x = model.predict(image)
        classes_x = np.argmax(predict_x, axis=1)[0]
        print("predicted sign: " + CLASSES_NAMES[classes_x] + " [" + str(classes_x) + "]   " + str(np.ndarray.max(predict_x)))

        if SHOW_IMAGE_AND_PREDICTION_IN_TESTING:
            plt.imshow(orig_img)
            plt.axis('off')
            plt.show()

def test(model, X_test, y_test):
    # REDO?
    """
    Test the model on dataset   
    """

    print(X_test.shape)
    # print(X_test)

    # print(predict_images.shape)

    for img in X_test:
        image = img.reshape(1, 32, 32, 1)
        print(image.shape)
        predicted_x = model.predict(image)
        predicted_class = np.argmax(predicted_x, axis=1)[0]

        print("predicted sign: " + CLASSES_NAMES[predicted_class] + \
            " [" + str(predicted_class) + "]   " + str(np.ndarray.max(predicted_x)))

        if SHOW_IMAGE_AND_PREDICTION_IN_TESTING:
            plt.imshow(img)
            plt.axis('off')
            plt.show()


def classify_frame(model, frame):
    """
    Classify a road sign from an image/frame
    Input: frame
    Return: predicted class, string of class, probability
    """

    img = preprocessing(frame)

    image = img.reshape(1, 32, 32, 1)
    
    predicted_x = model.predict(image)
    predicted_class = np.argmax(predicted_x, axis=1)[0]

    if SHOW_IMAGE_AND_PREDICTION_IN_TESTING:
        print("predicted sign: " + CLASSES_NAMES[predicted_class] + \
            " [" + str(predicted_class) + "]   " + str(np.ndarray.max(predicted_x)))
        plt.imshow(img)
        plt.imshow(frame)

        plt.axis('off')
        plt.show()


    return predicted_class, CLASSES_NAMES[predicted_class], np.ndarray.max(predicted_x)


def preprocessing(image):
    def grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


    def equalize(image):
        image = cv2.equalizeHist(image)
        return image

    image = grayscale(image)
    image = equalize(image)
    image = image/255
    return image

def postprocessing(X_train, X_valid, X_test, y_train, y_valid, y_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_valid = np.array(list(map(preprocessing, X_valid)))
    X_test = np.array(list(map(preprocessing, X_test)))


    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_valid = to_categorical(y_valid, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_TSC_model(model_name):
    model = load_model("models/" + model_name)
    return model


def build_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":


    # X_train, X_valid, X_test, y_train, y_valid, y_test = load_training_data()
    # print("Shapes of train data:")
    # print(X_train.shape, X_valid.shape, X_test.shape,
    #       y_train.shape, y_valid.shape, y_test.shape)

    # X_train, X_valid, X_test, y_train, y_valid, y_test = postprocessing(X_train, X_valid, X_test, y_train, y_valid, y_test)

    # if TRAINING_MODE:
    #     model = train(X_train, X_valid, y_train, y_valid)
    # else:
    #     model = load_model("models/" + MODEL_NAME)

    # if TESTING_MODE:
    #    test(model, X_test, y_test) 

    
    model = load_model("models/" + MODEL_NAME)
    path = os.getcwd() + "/data/predict/signs_1/"
    img_name = "image195-sign3.jpg"
    frame = imageio.v2.imread(path+img_name)
    classify_frame(model, frame)


    

    
