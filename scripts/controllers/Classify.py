from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

class Classify(object):

    clf = None

    def __init__(self):
        pass

    def classify(self, features, labels):
        self.clf = LinearSVC()
        self.clf.fit(features, labels)

    def predict(self, features):
        return self.clf.predict(features)

    def accuracy(self, features, labels):
        pred = self.predict(features)
        score = accuracy_score(labels, pred)
        return score

    def get_new_classifier(self):
        return LinearSVC()