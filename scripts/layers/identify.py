import cv2

from FeaturePipeline import FeaturePipeline
from layers.layer import Layer
import numpy as np

from models.featuremodel import FeatureModel
from pipeline import Pipeline


class Identify(Layer):

    def __init__(self):
        self.name = "Identify"
        self.pipeline = FeaturePipeline(name="Feature Pipeline", output_dir="../output_images/")
        self.pipeline.should_save_images = False
        pass

    def process(self, image_input, image_original, searchmodel, output_path):
        windows = searchmodel.windows
        on_windows = []

        for window in windows:
            test_img = cv2.resize(image_input[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            featuremodel = FeatureModel()
            self.pipeline.process(test_img, featuremodel)
            scaled_x = searchmodel.clf.normalizer.normalize_data(featuremodel)
            pred = searchmodel.clf.predict(scaled_x)
            if pred == 1:
                on_windows.append(window)

        searchmodel.positive_boxes = on_windows
        output_image = self.draw_boxes(image_input, on_windows)
        return image_input, output_image

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 255, 0), thick=4):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
