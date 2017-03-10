import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Trainer(object):

    train_data_path = ""
    split_ratio = 0.15
    random_state = 0

    train_features = []
    test_features = []

    train_labels = []
    test_labels = []

    def __init__(self, train_data_path, split_ratio = 0.15, random_state = 0):
        self.train_data_path = train_data_path
        self.split_ratio = split_ratio
        self.random_state = random_state
        pass

    # Load image data from a path, with a label. Join this into the class fields.
    def load_shuffle_split_join(self, data_path, data_label):
        images, labels = self.load_images_with_label(data_path, data_label)
        x_train, x_test, y_train, y_test = self.split_images(self.shuffle(images), labels)
        self.join_data(x_train, x_test, y_train, y_test)

    # Load images from a path, and assign them a label.
    def load_images_with_label(self, data_path, data_label):
        images = glob.glob(self.train_data_path + data_path + '*/*.png')
        labels = np.full((len(images)), data_label, dtype=np.int32)
        return images, labels

    # Shuffle the data.
    def shuffle(self, images):
        return shuffle(images, random_state=self.random_state)

    # Split the input into a training set and a validation set.
    def split_images(self, features, labels):
        train_set, test_set, train_labels, test_labels = \
            train_test_split(features, labels, test_size=self.split_ratio, random_state=self.random_state)
        return train_set, test_set, train_labels, test_labels

    # Join the new data into the existing class data.
    def join_data(self, train_set, test_set, train_labels, test_labels):
        self.train_features = np.concatenate((self.train_features, train_set))
        self.test_features = np.concatenate((self.test_features, test_set))
        self.train_labels = np.concatenate((self.train_labels, train_labels))
        self.test_labels = np.concatenate((self.test_labels, test_labels))



