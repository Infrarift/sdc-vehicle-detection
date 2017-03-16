from models.model import Model
import numpy as np

class StateModel(Model):

    def __init__(self):
        self.current_points = []
        self.current_id = 1
        pass

    def add_point(self, point_x, point_y, width):
        has_been_added = False
        for car_point in self.current_points:
            if car_point.is_in_range(point_x, point_y):
                car_point.add( point_x, point_y, width)
                has_been_added = True

        if has_been_added == False:
            self.current_points.append(CarPoint(self.current_id, point_x, point_y, width))
            self.current_id += 1
        pass

    def decay_points(self):
        new_array = []
        for car_point in self.current_points:
            car_point.decay_confidence(0.1)
            if car_point.confidence > 0:
                new_array.append(car_point)

        self.current_points = new_array

    def run_display(self):
        for car_point in self.current_points:
            car_point.run_display()

class CarPoint(object):

    search_range = 2000
    lerp_rate = 0.35
    display_lerp_rate = 0.35

    def __init__(self, id=0, x=0, y=0, width=70):
        self.confidence = 0.15
        self.x = x
        self.y = y
        self.display_x = x
        self.display_y = y
        self.id = id
        self.width = width
        self.display_width = width

    def add(self, point_x, point_y, width):
        self.x = self.lerp(self.x, point_x, self.lerp_rate)
        self.y = self.lerp(self.y, point_y, self.lerp_rate)
        self.width = self.lerp(self.width, width, self.lerp_rate)
        self.confidence += 0.15
        if self.confidence > 1:
            self.confidence = 1

    def decay_confidence(self, value):
        self.confidence -= value
        if self.confidence <= 0:
            self.confidence = 0

    def run_display(self):
        self.display_x = self.lerp(self.display_x, self.x, self.display_lerp_rate)
        self.display_y = self.lerp(self.display_y, self.y, self.display_lerp_rate)
        self.display_width = self.lerp(self.display_width, self.width, self.display_lerp_rate)

    def is_in_range(self, point_x, point_y):
        x_len = self.x - point_x
        y_len = self.y - point_y
        distance_squared = x_len ** 2 + y_len ** 2
        return distance_squared < self.search_range

    def lerp(self, a, b, f):
        return a + f * (b - a)