import cv2

from layers.layer import Layer

class SpatialBin(Layer):

    target_size = (32, 32)

    def __init__(self, target_size = (32, 32)):
        self.name = "Spatial Bin"
        self.target_size = target_size
        pass

    def process(self, image_input, image_original, model, output_path):
        output_image = cv2.resize(image_input, self.target_size)
        return output_image