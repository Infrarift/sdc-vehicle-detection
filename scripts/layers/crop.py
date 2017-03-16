from layers.layer import Layer

class Crop(Layer):

    def __init__(self, top_crop_ratio = 0, bot_crop_ratio = 0, side_crop_ratio= 0):
        self.name = "Crop"
        self.top_crop_ratio = top_crop_ratio
        self.bot_crop_ratio = bot_crop_ratio
        self.side_crop_ratio = side_crop_ratio
        pass

    def process(self, image_input, image_original, featuremodel, output_path):

        y_size = image_input.shape[0]
        x_size = image_input.shape[1]

        y_start = int(self.top_crop_ratio * y_size)
        y_stop = int((1-self.bot_crop_ratio) * y_size)

        x_start = int(self.side_crop_ratio * x_size)
        x_stop = int((1 - self.side_crop_ratio) * x_size)

        featuremodel.crop_offset = (x_start, y_start)

        output_image = image_input[y_start:y_stop,x_start:x_stop,:]
        return output_image, output_image