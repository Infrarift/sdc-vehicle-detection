from skimage.feature import hog
from layers.layer import Layer
import cv2

class Hog(Layer):

    orientations = 6
    px_per_cell = 4
    cl_per_block = 2

    def __init__(self, orientations = 6, px_per_cell = 4, cl_per_block = 2):
        self.name = "HOG"
        self.orientations = orientations
        self.px_per_cell = px_per_cell
        self.cl_per_block = cl_per_block
        pass

    def process(self, image_input, image_original, featuremodel, output_path):

        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)

        if self.should_save:
            features, hog_image = hog(gray,
                                orientations=self.orientations,
                                pixels_per_cell=(self.px_per_cell, self.px_per_cell),
                                cells_per_block=(self.cl_per_block, self.cl_per_block),
                                transform_sqrt=False,
                                visualise=True,
                                feature_vector=True)
            image_input = hog_image

        else:
            features = hog(gray,
                                orientations=self.orientations,
                                pixels_per_cell=(self.px_per_cell, self.px_per_cell),
                                cells_per_block=(self.cl_per_block, self.cl_per_block),
                                transform_sqrt=False,
                                visualise=False,
                                feature_vector=True)

        featuremodel.add_to_current_feature(features)
        return image_input