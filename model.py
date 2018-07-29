import os
import csv

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout, Activation

EPOCHS = 1
OFF_SIDE_CORRECTION_FACTOR = 1.3
DRIVING_WELL_LABELS = [
    "./raw_collected_data/driving_log.csv",
    "./raw_collected_bridge_end/driving_log.csv",
]

def fetch_labeled_data_from_disk(collection_path):
    """
    Fetch images and labels from disk given a path. NO GENERATOR IS USED. The memory benefit of using a generator
    is irrelevant for this project given how few labels I used.
    """
    lines = []
    with open (collection_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        if not os.path.exists(line[0]):
            # Support running on my local machine and my remote machine.
            line[0] = line[0].replace("/Users/attwell/Uber/Github/", "/home/attwell/GitHub/")
            line[0] = line[0].replace("/Users/attwell/Uber/GitHub/", "/home/attwell/GitHub/")
        inner_path = line[0]
        if not os.path.exists(inner_path):
            raise Exception("Path doesn't exist. Path=" + inner_path)
        image = cv2.imread(inner_path)
        images.append(image)
        measurements.append(float(line[3]))
    x_data = images
    y_data = measurements
    return x_data, y_data

def flip_labelled_data(x_data, y_data):
    """
    Create a mirrored version of the input data. The images are mirrored and the labels are multiplied by -1.
    """
    new_x_data = []
    new_y_data = []
    for index, _ in enumerate(x_data):
        image = x_data[index]
        new_x_data.append(cv2.flip(image, 1))
        new_y_data.append(-y_data[index])
    return new_x_data, new_y_data


def load_too_far_to_size_labelled_data(adjustment_factor, far_right_path):
    """
    Load input images associated with driving too far on one side of the road.
    """
    # load images and associate it with driving too far on the right (or left if -ve adjustment factor)
    x_data1, y_data1 = fetch_labeled_data_from_disk(far_right_path)
    for index, _ in enumerate(y_data1):
        y_data1[index] = -adjustment_factor
    # mirror the data and associate it with driving too far on the left (or right if -ve adjustment factor)
    x_data2, y_data2 = fetch_labeled_data_from_disk(far_right_path)
    x_data2, y_data2 = flip_labelled_data(x_data2, x_data2)
    for index, _ in enumerate(y_data2):
        y_data2[index] = adjustment_factor
    return x_data1 + x_data2, y_data1 + y_data2


def create_model(x_data, y_data):
    """
    Train the model.
    """
    model = Sequential()
    # Crop irrelevant parts out.  "If tuple of 2 tuples of 2 ints:
    # interpreted as  ((top_crop, bottom_crop), (left_crop, right_crop))"
    model.add(Cropping2D(cropping=((60, 30), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize the pixel values between -1 and 1.
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(6, 10, 10, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 10, 10, subsample=(2, 2), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    # by default this trains for 10 epochs.
    model.fit(x_data, y_data, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)
    return model


# Labels for showing the car how to drive well.
x_data = []
y_data = []
for path in DRIVING_WELL_LABELS:
    x_data1, y_data1 = fetch_labeled_data_from_disk(path)
    x_data2, y_data2 = flip_labelled_data(x_data1, y_data1)
    x_data = x_data2 + x_data1 + x_data
    y_data = y_data2 + y_data1 + y_data

# Labels for showing the car how to recover from going off the side of the road.
x_data_2, y_data_2 = load_too_far_to_size_labelled_data(OFF_SIDE_CORRECTION_FACTOR,
                                                        "./raw_collected_drive_right/driving_log.csv")
x_data_3, y_data_3 = load_too_far_to_size_labelled_data(-OFF_SIDE_CORRECTION_FACTOR,
                                                        "./raw_collected_drive_left/driving_log.csv")
x_data = np.array(x_data + x_data_2 + x_data_3)
y_data = np.array(y_data + y_data_2 + y_data_3)

model = create_model(x_data, y_data)
model.save("model.h5")
