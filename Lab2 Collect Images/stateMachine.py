'''
Runs the state machine for Lab 2
'''

import sys
import os
import time
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import datetime
import time
import asyncio

import numpy as np

import pickle

from skimage import color

from imgclassification import ImageClassifier

def run(sdk_conn):

    robot = sdk_conn.wait_for_robot()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    img_clf = None

    if (os.path.isfile('classifier.p')):
        print("Pre-trained model found")
        img_clf = pickle.load(open('classifier.p', 'rb'))
    else:
        print("Pre-trained model not found - Begin training")
        img_clf = getTrainedModel()
        pickle.dump(img_clf, open('classifier.p', 'wb'))
        print("Training done")


    state = 'idle'

    while(True):
        if (state == 'idle'):
            latest_image = robot.world.latest_image
            new_image = latest_image.raw_image

            predicted_label = img_clf.predict_labels(img_clf.extract_image_features(np.array(new_image).reshape( (-1,240,320,3) )))[0]

            if (predicted_label != 'none'):
                print(predicted_label)
                robot.say_text(predicted_label).wait_for_completed()

            if (predicted_label == 'drone'):
                state = 'drone'
            if (predicted_label == 'order'):
                state = 'order'
            if (predicted_label == 'inspection'):
                state = 'inspection'

        if (state == 'drone'):
            look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            cube = None
            try:
                cube = robot.world.wait_for_observed_light_cube(timeout=20)
            except asyncio.TimeoutError:
                print("There is no cube")
            finally:
                look_around.stop()
            if cube is not None:
                robot.go_to_object(cube, distance_mm(40)).wait_for_completed()
                robot.pickup_object(cube, num_retries = 4).wait_for_completed()
                robot.drive_straight(distance_mm(100), speed_mmps(30)).wait_for_completed()
                robot.place_object_on_ground_here(cube).wait_for_completed()
                robot.drive_straight(distance_mm(-100), speed_mmps(30)).wait_for_completed()

            state = 'idle'

        if (state == 'order'):
            robot.drive_wheels(50,23.5, duration = 21.55)

            state = 'idle'

        if (state == 'inspection'):
            robot.set_lift_height(0,duration = 1, in_parallel = True).wait_for_completed()
            for times in range(4):
                drive_action = robot.drive_straight(distance_mm(200), speed_mmps(50), in_parallel = True)

                robot.set_lift_height(1,duration = 2, in_parallel = True).wait_for_completed()
                robot.set_lift_height(0,duration = 2, in_parallel = True).wait_for_completed()

                drive_action.wait_for_completed()

                robot.turn_in_place(degrees(90)).wait_for_completed()

            state = 'idle'


def getTrainedModel():
    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)

    return img_clf

if __name__ == '__main__':
    cozmo.setup_basic_logging()

    try:
        cozmo.connect(run)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
