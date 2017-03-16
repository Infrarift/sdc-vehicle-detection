from models.model import Model

class SearchModel(Model):

    def __init__(self):
        self.windows = None
        self.clf = None
        self.state_model = None