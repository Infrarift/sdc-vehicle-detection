class Layer(object):

    name = "Base"
    should_save = True
    override_original = False  # If true, this output will override "original image" in the pipeline.
    id = 0

    def __init__(self):
        self.name = "Base"


    def process(self, image_input, image_original, model, output_path):
        return image_input, image_input