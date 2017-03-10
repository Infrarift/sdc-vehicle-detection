import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class Normalize(object):

    def normalize_data(self, featuremodel):
        features = np.vstack(featuremodel.features_raw).astype(np.float64)
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        featuremodel.features_scaled = scaled_features
        return scaled_features

    def visualize(self, images, output_path, features, scaled_features):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(0, len(images)):
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(features[i])
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_features[i])
            plt.title('Normalized Features')
            fig.tight_layout()
            plt.savefig(output_path + "plot{0}.png".format(i))
            plt.clf()