import cv2
from layers.layer import Layer
import numpy as np
import matplotlib.pyplot as plt

class Heatmap(Layer):

    def __init__(self):
        self.name = "Heatmap"
        pass

    def process(self, image_input, image_original, featuremodel, output_path):

        heat_map = np.full((image_input.shape[0], image_input.shape[1]), 0, dtype=np.int32)
        heat_map = self.add_heat(heat_map, featuremodel.positive_boxes)
        heat_map = self.apply_threshold(heat_map, 3)
        overlay_map = np.dstack((heat_map * 0, heat_map * 5, heat_map * 25))
        overlay_map = np.clip(overlay_map, 0, 255)
        output_image = self.overlay(image_input, overlay_map)
        featuremodel.heat_map = heat_map
        return image_input, output_image

    def add_heat(self, heatmap, box_list):
        for box in box_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def overlay(self, image_base, image_overlay):
        alpha = 1
        beta = 0.8
        dst = image_base * alpha + image_overlay * beta;
        dst = np.clip(dst, 0, 255)
        return dst

