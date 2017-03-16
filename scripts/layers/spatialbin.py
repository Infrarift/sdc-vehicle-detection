import cv2

from layers.layer import Layer

class SpatialBin(Layer):


    def __init__(self, target_size = (32, 32)):
        self.name = "Spatial Bin"
        self.target_size = target_size
        pass

    def process(self, image_input, image_original, featuremodel, output_path):
        output_image = cv2.resize(image_input, self.target_size)
        feature = output_image.ravel()
        featuremodel.add_to_current_feature(feature)
        return output_image, output_image