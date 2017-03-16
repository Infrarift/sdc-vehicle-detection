import cv2

from layers.layer import Layer
import numpy as np
from math import ceil

class Windows(Layer):


    def __init__(self, top_crop_ratio = 0, bot_crop_ratio = 0, side_crop_ratio= 0):
        self.name = "Windows"
        pass

    def process(self, image_input, image_original, searchmodel, output_path):
        window_list = self.slide_window(image_input)
        output_image = self.draw_boxes(image_input, window_list)
        searchmodel.windows = window_list
        return image_input, output_image

    def slide_window(self, img):

        # Compute the span of the region to be searched
        xspan = img.shape[1]
        yspan = img.shape[0]

        window_ratio = 1.65
        window_start_h = 65

        x_desired_overlap = 0.85
        y_desired_overlap = 0.85

        window_scale_from_pos_factor = 0 # At which y-pos in the image to begin scaling up.
        window_size_max_factor = 2.5 # Percent of original size it will grow to.

        window_scale_from_pos = yspan * window_scale_from_pos_factor
        window_scale_range = yspan - window_scale_from_pos
        window_end_h = int(window_start_h * window_size_max_factor)
        x_overlap_recipricol = 1 - x_desired_overlap
        y_overlap_recipricol = 1 - y_desired_overlap

        window_list = []

        process_complete = False
        y_start = 0
        while not process_complete:

            # Find out the height from the current y start
            y_height = self.calculate_window_height(window_end_h,
                                                    window_scale_from_pos,
                                                    window_scale_range,
                                                    window_start_h,
                                                    y_start)

            # Find the window's end y pos.
            y_end = y_start + y_height

            # Check if we overflow
            y_overflow = y_end - yspan
            if y_overflow >= 0:
                # End the loop
                process_complete = True
                y_height = window_end_h
                y_start = yspan - window_end_h
                y_end = yspan

            # Calculate the window's width.
            x_width = y_height * window_ratio

            # Find number of x windows
            number_of_x_windows = int(ceil(xspan / (x_width * x_overlap_recipricol)))
            # Find best overlap value for the search.
            x_overlap = self.find_best_x_overlap(number_of_x_windows, x_overlap_recipricol, x_width, xspan)
            x_start_step = x_width * (1 - x_overlap)


            # Add the windows
            self.add_row_windows(number_of_x_windows, window_list, x_start_step, x_width, y_end, y_start)

            # Find next starting position
            y_start += int(y_height * y_overlap_recipricol)


        return window_list

    def add_row_windows(self, number_of_x_windows, window_list, x_start_step, x_width, y_end, y_start):
        for nx in range(0, number_of_x_windows):
            x_start = int(nx * x_start_step)
            x_end = int(x_start + x_width)
            window_list.append(((x_start, y_start), (x_end, y_end)))

    def find_best_x_overlap(self, number_of_x_windows, x_overlap_recipricol, x_width, xspan):
        window_cumulative_span = x_width * number_of_x_windows
        best_overlap = (xspan - window_cumulative_span) / -window_cumulative_span
        return best_overlap

    def calculate_window_height(self, window_end_h, window_scale_from_pos, window_scale_range, window_start_h, y_start):
        h_adjusted = y_start - window_scale_from_pos
        h_progress = h_adjusted / window_scale_range
        if h_progress <= 0:
            return window_start_h
        return int(self.lerp(window_start_h, window_end_h, h_progress))

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=2):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def lerp(self, a, b, f):
        return a + f * (b - a)