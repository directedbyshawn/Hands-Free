import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import os

import imageio
import glob
import cv2

import pandas as pd
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Questions:
# train on rgb or grey pictures?
# now i can predict single sign, how can i predicte multiple signs in the picture like yolo??? maybe sliding window, ancher boxes
# how to convert 3d greyscale picture to 2d greyscale

# My plans:
# prework: delete garbage pictures
# 1. Classify only Stop signs
# 2. Classify all available signs for 1 picture-sign
# 3. Classify signs on the real image situation

signs_dict = {"classes": ["stop", "speedLimitUrdbl", "speedLimit25", "pedestrianCrossing", "speedLimit35", "turnLeft", "slow", "speedLimit15", "speedLimit45", "rightLaneMustTurn", "signalAhead", "keepRight", "laneEnds", "school", "merge", "addedLane", "rampSpeedAdvisory40", "rampSpeedAdvisory45", "curveRight", "speedLimit65", "truckSpeedLimit55", "thruMergeLeft", "speedLimit30", "stopAhead", "yield", "thruMergeRight", "dip", "schoolSpeedLimit25", "thruTrafficMergeLeft", "noRightTurn", "rampSpeedAdvisory35", "curveLeft", "rampSpeedAdvisory20", "noLeftTurn", "zoneAhead25", "zoneAhead45", "doNotEnter", "yieldAhead", "roundabout", "turnRight", "speedLimit50", "rampSpeedAdvisoryUrdbl", "rampSpeedAdvisory50", "speedLimit40", "speedLimit55", "doNotPass", "intersection"], "name_to_label": {"stop": 0, "speedLimitUrdbl": 1, "speedLimit25": 2, "pedestrianCrossing": 3, "speedLimit35": 4, "turnLeft": 5, "slow": 6, "speedLimit15": 7, "speedLimit45": 8, "rightLaneMustTurn": 9, "signalAhead": 10, "keepRight": 11, "laneEnds": 12, "school": 13, "merge": 14, "addedLane": 15, "rampSpeedAdvisory40": 16, "rampSpeedAdvisory45": 17, "curveRight": 18, "speedLimit65": 19, "truckSpeedLimit55": 20, "thruMergeLeft": 21, "speedLimit30": 22, "stopAhead": 23, "yield": 24, "thruMergeRight": 25, "dip": 26, "schoolSpeedLimit25": 27, "thruTrafficMergeLeft": 28, "noRightTurn": 29, "rampSpeedAdvisory35": 30, "curveLeft": 31, "rampSpeedAdvisory20": 32, "noLeftTurn": 33, "zoneAhead25": 34, "zoneAhead45": 35, "doNotEnter": 36, "yieldAhead": 37, "roundabout": 38, "turnRight": 39, "speedLimit50": 40, "rampSpeedAdvisoryUrdbl": 41, "rampSpeedAdvisory50": 42, "speedLimit40": 43, "speedLimit55": 44, "doNotPass": 45, "intersection": 46}}

def save_imgs_from_tensor():
    path = ".../ML_Project/lisa/data/lisa-batches/"
    name = "images_0.tensor"
    # name = "images_1.tensor"
    # name = "images_2.tensor"

    label_name = "labels.tensor"

    labels = torch.load(path+label_name)

    images = torch.load(path+name)
    # print(images.shape)
    # print(len(labels))
    # print(len(images))

    for i, img in zip(range(0, len(images)), images):
        # print(i, img.shape, int(labels[i]))
        print('data/copy/' + str(int(labels[i]))+ '/' + str(i) + '.png')
        torchvision.utils.save_image(img, 'data/copy/' + str(int(labels[i]))+ '/' + str(i) + '.png')


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

    path = os.getcwd() + "/data/predict/"

    for im_path in glob.glob(path + "*.png"):
        image = imageio.v2.imread(im_path)

        images.append(image)

    return images


def test_load():
    train = []
    test = []
    for n_class in range(0, 43):#[0, 2, 3, 4, 5, 6, 10, 11, 12, 14]:

        path = os.getcwd() + "/data/" + str(n_class) + "/"
        # print(path)

        class_train = []
        for im_path in glob.glob(path + "*.png"):
            image = imageio.v2.imread(im_path)

            # print(image.shape)
            class_train.append(image)

        # to "normalize" the number of each class
        if len(class_train) > 150:
            np.random.shuffle(class_train)
            class_train = class_train[:150][:][:]

        train.extend(class_train)
        test.extend([n_class] * len(class_train))


    train = np.array(train)
    test = np.array(test)

    # print(train.shape)
    # print(test.shape)

    X_train, X_valid, X_test, y_train, y_valid, y_test = divide_data(train, test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def predict_on_data_from_predict_folder():
    predict_images = load_predict()
    for orig_img in predict_images:
        img = preprocessing(orig_img)
        image = img.reshape(1, 32, 32, 1)

        predict_x = model.predict(image)
        classes_x = np.argmax(predict_x, axis=1)[0]
        print("predicted sign: " + classes_names[classes_x] + " [" + str(classes_x) + "]   " + str(np.ndarray.max(predict_x)))
        plt.imshow(orig_img)
        plt.axis('off')
        plt.show()

def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def equalize(image):
    image = cv2.equalizeHist(image)
    return image


def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    image = image/255
    return image


def modified_model():
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
    model.add(Dense(43, activation='softmax'))

    # Compile Model
    model.compile(Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def map_argmax(predict_x):
    return np.argmax(predict_x, axis=1)[0]

if __name__ == "__main__":

    classes_names = signs_dict["classes"]

    # exit()

    X_train, X_valid, X_test, y_train, y_valid, y_test = test_load()
    print(X_train.shape, X_valid.shape, X_test.shape,
          y_train.shape, y_valid.shape, y_test.shape)

    X_train = np.array(list(map(preprocessing, X_train)))
    X_valid = np.array(list(map(preprocessing, X_valid)))
    X_test = np.array(list(map(preprocessing, X_test)))

    # print(X_train.shape[0])

    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

    y_train = to_categorical(y_train, 43)
    y_valid = to_categorical(y_valid, 43)
    y_test = to_categorical(y_test, 43)

    save_model = False
    model_to_load = True
    model_name = "greyScaleEqualizedWithValidation_43_all_data.h5"

    if model_to_load:
        model = load_model("cfg/" + model_name)
    else:
        model = modified_model()
        print(model.summary())
        # history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=50),
        #                           steps_per_epoch=X_train.shape[0]/50,
        #                           epochs=15,
        #                           validation_data=(X_valid, y_valid),
        #                           shuffle = 1)\

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=10)
        print(history)
    
        if save_model:
            model.save("cfg/" + model_name)

    

    # print(X_test.shape)
    predict_x = model.predict(X_test)
    # print(predict_x.shape)
    y_pred = []
    for y in predict_x:
        # print(y)
        pred = np.argmax(y)
        # print(pred)
        y_pred.append(pred)
    
    y_my_test = []
    for y in y_test:
        # print(y)
        pred = np.argmax(y)
        # print(pred)
        y_my_test.append(pred)

    y_pred = np.array(y_pred)
    y_test = np.array(y_my_test)

    prfs = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    print("prfs ", prfs)

    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix.shape)
    # print(conf_matrix[:5,:])
    # print()
    # print(conf_matrix[-5:,:])

    # exit()
    # for i in range(0, len(X_test), 7):

    #     image = X_test[i].reshape(1, 32, 32, 1)

    #     predict_x = model.predict(image)
    #     classes_x = np.argmax(predict_x, axis=1)[0]
    #     print("predicted sign: " + classes_names[classes_x] + " [" + str(classes_x) + "]")
    #     plt.imshow(X_test[i])
    #     plt.axis('off')
    #     plt.show()

    # conf_matrix = confusion_matrix(y_true, y_pred)

    predict_on_data_from_predict_folder()
