from layers.layer import Layer

class Combine(Layer):

    def __init__(self):
        self.name = "Combine"
        self.should_save = False
        pass

    def process(self, image_input, image_original, featuremodel, output_path):
        featuremodel.add_current_feature_to_vectors()
        featuremodel.clear_current_feature()
        return image_input, image_input

