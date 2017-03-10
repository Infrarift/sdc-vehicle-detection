from cv2 import imwrite
import os

class Pipeline(object):

    layers = []
    dir_path = ""
    current_image_id = 0
    output_step = 1
    max_save_images = 25
    should_save_images = True

    def __init__(self, name = "Pipeline", output_dir = "../output_images", output_step = 1):
        self.dir_path = output_dir + "/" + name + "/"
        self.output_step = output_step

    def add_layer(self, layer):
        layer.id = len(self.layers)
        self.layers.append(layer)

    def process(self, image, model):

        self.increment_id(model)
        print("Processing ", self.current_image_id)
        output_image = image

        for layer in self.layers:
            self.set_layer_saving(layer)
            output_image = layer.process(output_image, image, model, self.output_path_of_layer(layer))
            if layer.override_original:
                image = output_image
            self.save_image(output_image, layer)

        return output_image

    def set_layer_saving(self, layer):
        if self.can_save():
            self.create_output_dir(layer)
            layer.should_save = self.can_save_on_frame(self.current_image_id, self.output_step)
        else:
            layer.should_save = False

    def increment_id(self, model):
        self.current_image_id += 1
        model.current_image_id = self.current_image_id

    def can_save_on_frame(self, frame_id, step):
        return frame_id % step == 0

    def create_output_dir(self, layer):
        if layer.should_save:
            final_dir = self.output_path_of_layer(layer)
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)

    def save_image(self, image, layer):
        if layer.should_save and self.can_save():
            img_name = "image_" + str(self.current_image_id) + ".jpg"
            final_dir = self.output_path_of_layer(layer)
            imwrite(final_dir + img_name, image);

    def output_path_of_layer(self, layer):
        return "{0}Layer{1}_{2}/".format(self.dir_path, layer.id, layer.name)

    def output_path(self, name="Unnamed"):
        return "{0}Layer{1}_{2}/".format(self.dir_path, "X", name)

    def can_save(self):
        if not self.should_save_images:
            return False
        if (self.current_image_id >= self.max_save_images) and (self.max_save_images != 0):
            return False
        return True