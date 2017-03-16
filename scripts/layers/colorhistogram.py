import cv2
from layers.layer import Layer
import numpy as np
import matplotlib.pyplot as plt

class ColorHistogram(Layer):

    def __init__(self):
        self.name = "Color Histogram"
        pass

    def process(self, image_input, image_original, featuremodel, output_path):

        image = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)

        hist_0 = np.histogram(image[:, :, 0], bins=32, range=(0, 256))
        hist_1 = np.histogram(image[:, :, 1], bins=32, range=(0, 256))
        hist_2 = np.histogram(image[:, :, 2], bins=32, range=(0, 256))

        self.visualize(hist_0, hist_1, hist_2, output_path, featuremodel.current_image_id,
                       h1_name="Feature 1",
                       h2_name="Feature 2",
                       h3_name="Feature 3")

        featuremodel.add_to_current_feature(hist_0[0])
        featuremodel.add_to_current_feature(hist_1[0])
        featuremodel.add_to_current_feature(hist_2[0])
        return image_input, image_input

    def visualize(self, hist_0, hist_1, hist_2, output_path, id,
                  h1_name = "Feature 1",
                  h2_name = "Feature 2",
                  h3_name = "Feature 3"):

        if not self.should_save:
            return

        # Generating bin centers
        bin_edges = hist_0[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Plot a figure with all three bar charts
        fig = plt.figure(figsize=(12, 3))
        self.plot_graph(bin_centers, h1_name, hist_0, 131)
        self.plot_graph(bin_centers, h2_name, hist_1, 132)
        self.plot_graph(bin_centers, h3_name, hist_2, 133)
        plt.title(h3_name)

        if self.should_save:
            plt.savefig(output_path + "plot{0}.png".format(id))
            plt.clf()
            plt.close()

    def plot_graph(self, bin_centers, h_name, hist, subplot_id):
        plt.subplot(subplot_id)
        plt.bar(bin_centers, hist[0])
        plt.xlim(0, 256)
        plt.title(h_name)
