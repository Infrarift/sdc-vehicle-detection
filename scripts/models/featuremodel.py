from models.model import Model
import numpy as np

class FeatureModel(Model):
    def __init__(self):
        self.features_raw = []
        self.features_scaled = None
        self.labels = []
        self.current_feature = None
        self.positive_boxes = None
        self.heat_map = None
        self.crop_offset = None
        self.state_model = None
        pass

    def clear_feature_vector(self):
        self.features_raw = []

    def clear_current_feature(self):
        self.current_feature = None

    def add_current_feature_to_vectors(self):
        self.features_raw.append(self.current_feature)

    def add_to_current_feature(self, vector):
        if self.current_feature is None:
            self.current_feature = vector
        else:
            self.current_feature = np.concatenate((self.current_feature, vector))

    def add_label(self, label):
        self.labels.append(label)

