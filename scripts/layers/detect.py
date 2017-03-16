import cv2
from layers.layer import Layer
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

class Detect(Layer):

    def __init__(self):
        self.name = "Detect"
        self.x_offset = 0
        self.y_offset = 0
        pass

    def process(self, image_input, image_original, searchmodel, output_path):

        self.x_offset = searchmodel.crop_offset[0]
        self.y_offset = searchmodel.crop_offset[1]

        labels = label(searchmodel.heat_map)
        output_image = self.draw_labeled_bboxes(image_original, labels, searchmodel)
        searchmodel.state_model.run_display()
        searchmodel.state_model.decay_points()
        return output_image, output_image

    def draw_labeled_bboxes(self, img, labels, searchmodel):

        for car_number in range(1, labels[1] + 1):

            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            size_x = abs(bbox[0][0] - bbox[1][0])
            size_y = abs(bbox[0][1] - bbox[1][1])
            min_box_size = 30


            if (size_x >= min_box_size) and (size_y >= min_box_size):
                center_point = self.get_center(bbox[0], bbox[1])
                searchmodel.state_model.add_point(center_point[0], center_point[1], size_x)

        for car_point in searchmodel.state_model.current_points:
            center_point = (int(car_point.display_x), int(car_point.display_y))
            self.register_car_at_point(img, center_point, car_point.confidence, car_point.id, car_point.display_width)

        return img

    def get_center(self, top, bottom):
        return (int(self.lerp(top[0], bottom[0], 0.5)) + self.x_offset, int(self.lerp(top[1], bottom[1], 0.5) + self.y_offset))

    def register_car_at_point(self, image, center_point, confidence, id, width):
        if confidence > 0.35:
            self.draw_cross_at_point(image, center_point, confidence, id)
            self.draw_rect_at_point(image, center_point, confidence, id, width)
        pass

    def draw_cross_at_point(self, image, point, confidence, id):

        line_size = 10
        line_color = self.get_box_color(confidence)
        line_thick = 2

        p_x = int(point[0])
        p_y = int(point[1])

        p1 = (p_x - line_size, p_y)
        p2 = (p_x + line_size, p_y)
        p3 = (p_x, p_y - line_size)
        p4 = (p_x, p_y + line_size)

        cv2.line(image, p1, p2, line_color, line_thick)
        cv2.line(image, p3, p4, line_color, line_thick)

        pass

    def draw_rect_at_point(self, image, point, confidence, id, x_size = 70):

        rect_size_x = int(x_size/2)
        rect_size_ratio = 0.6

        rect_size_y = int(rect_size_x * rect_size_ratio)


        p1 = (point[0] - rect_size_x, point[1] - rect_size_y)
        p2 = (point[0] + rect_size_x, point[1] + rect_size_y)
        p3 = (p1[0], p1[1] - 10)

        cv2.rectangle(image, p1, p2, self.get_box_color(confidence), 2)
        self.draw_text(image, "V{0}: {1}".format(id, self.conf_as_percent(confidence)), p3, (rect_size_x * 2, 24), self.get_box_color(confidence))

    def get_box_color(self, confidence):
        if confidence < 0.5:
            return (0, 0, 255)
        elif confidence < 0.85:
            return (0, 150, 255)
        else:
            return (0, 255, 50)

    def conf_as_percent(self, confidence):
        return "{0}%".format(int(confidence * 100))

    def draw_text(self, image, text, position, size, color = [255, 255, 255]):
        r_height = size[1]
        r_width = size[0]
        r_padding = 4
        r_pos1 = (position[0], position[1] + r_padding)
        r_pos2 = (position[0] + r_width, position[1] - r_height)
        t_pos = (position[0] + r_padding, position[1] - r_padding)
        image = cv2.rectangle(image, r_pos1, r_pos2, (0, 0, 0), thickness=-1)
        image = cv2.putText(image, text, t_pos, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        return image

    def lerp(self, a, b, f):
        return a + f * (b - a)



