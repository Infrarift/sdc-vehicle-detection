import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Trainer(object):

    train_data_path = ""
    split_ratio = 0.15
    random_state = 0



    def __init__(self, train_data_path, split_ratio = 0.15, random_state = 0):
        self.train_data_path = train_data_path
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.features = []
        self.labels = []
        pass

    # Load image data from a path, with a label. Join this into the class fields.
    def load_shuffle_split_join(self, data_path, data_label):
        features, labels = self.load_images_with_label(data_path, data_label)
        self.join_data(features, labels)

    # Load images from a path, and assign them a label.
    def load_images_with_label(self, data_path, data_label):
        images = glob.glob(self.train_data_path + data_path + '*/*.png')
        labels = np.full((len(images)), data_label, dtype=np.int32)
        return images, labels

    # Shuffle the data.
    def shuffle(self, images):
        return shuffle(images, random_state=self.random_state)

    # Join the new data into the existing class data.
    def join_data(self, features, labels):
        self.features = np.concatenate((self.features, features))
        self.labels = np.concatenate((self.labels, labels))
        self.features = shuffle(self.features, random_state=self.random_state)
        self.labels = shuffle(self.labels, random_state=self.random_state)




