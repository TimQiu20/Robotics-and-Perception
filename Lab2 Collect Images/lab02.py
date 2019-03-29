import sys
import cozmo
import datetime
import time
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import asyncio

def cozmo_idle(robot:cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()

    img_clf = ImageClassifier()

    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    train_data = img_clf.extract_image_features(train_raw)

    img_clf.train_classifier(train_data, train_labels)

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    while True:
        time.sleep(1.0)
        latest_im = robot.world.latest_image
        new_im = np.asarray(latest_im.raw_image, dtype = "uint8")
        f = feature.hog(new_im, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4), feature_vector=True,
                        block_norm='L2-Hys')
        type = img_clf.predict_labels([f])[10]

        if type == "drone":
            robot.say_text(type).wait_for_completed()
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
                robot.pickup_object(cube, num_retries = 2).wait_for_completed()
                robot.drive_straight(distance_mm(100), speed_mmps(55)).wait_for_completed()
                robot.place_object_on_ground_here(cube).wait_for_completed()
                robot.drive_straight(distance_mm(-100), speed_mmps(55)).wait_for_completed()

        elif type == "order":
            robot.say_text(type).wait_for_completed()
            robot.drive_wheels(50,23.5, duration = 21.55)

        elif type == "inspection":
            robot.say_text(type).wait_for_completed()
            for times in range(4):
                robot.set_lift_height(1,duration = 2, in_parallel = True)
                robot.drive_straight(distance_mm(200), speed_mmps(50), in_parallel = True).wait_for_completed()
                robot.set_lift_height(0,duration = 2, in_parallel = True).wait_for_completed()
                robot.turn_in_place(degrees(90)).wait_for_completed()

        elif type == "plane":
            robot.say_text(type).wait_for_completed()

        elif type == "truck":
            robot.say_text(type).wait_for_completed()

        elif type == "hands":
            robot.say_text(type).wait_for_completed()

        elif type == "place":
            robot.say_text(type).wait_for_completed()

        time.sleep(1)


class ImageClassifier:

    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir + "*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return (data, labels)

    def extract_image_features(self, data):
        l = []
        for im in data:
            im_gray = color.rgb2gray(im)

            im_gray = filters.gaussian(im_gray, sigma=0.4)

            f = feature.hog(im_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4),
                            feature_vector=True, block_norm='L2-Hys')
            l.append(f)

        feature_data = np.array(l)
        return (feature_data)

    def train_classifier(self, train_data, train_labels):
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifer.predict(data)
        return predicted_labels


cozmo.run_program(cozmo_idle)
