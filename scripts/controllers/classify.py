from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

class Classify(object):

    clf = None
    normalizer = None

    def __init__(self):
        pass

    def classify(self, features, labels):
        self.clf = LinearSVC(max_iter=1000, C=2.0)
        self.clf.fit(features, labels)

    def predict(self, features):
        return self.clf.predict(features)

    def accuracy(self, features, labels):
        pred = self.predict(features)
        score = accuracy_score(labels, pred)
        return score

    def get_new_classifier(self):
        return LinearSVC()