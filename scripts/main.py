from cv2 import imread

from sklearn.model_selection import train_test_split
from controllers.Classify import Classify
from controllers.normalize import Normalize
from layers.colorhistogram import ColorHistogram
from layers.combine import Combine
from layers.hog import Hog
from layers.spatialbin import SpatialBin
from models.featuremodel import FeatureModel
from pipeline import Pipeline
from trainer import Trainer

def load_training_data():
    training_path = "../training_data/"
    trainer = Trainer(training_path, split_ratio=0)
    trainer.load_shuffle_split_join("vehicles/", 1)
    trainer.load_shuffle_split_join("non-vehicles/", 0)
    print("Data loading complete: {0} Training Features, {1} Testing Features"
          .format(len(trainer.train_features), format(len(trainer.test_features))))
    print(trainer.train_features[0])
    return trainer

def run_test_pipeline():

    pipeline = Pipeline(name="Test Pipeline", output_dir="../output_images/")
    pipeline.add_layer(SpatialBin())
    pipeline.add_layer(ColorHistogram())
    pipeline.add_layer(Hog())
    pipeline.add_layer(Combine())

    model = FeatureModel()
    trainer = load_training_data()

    visual_images = []
    visual_count = 25

    for i in range(0, len(trainer.train_features)):
        image_path = trainer.train_features[i]
        label = trainer.train_labels[i]
        model.add_label(label)
        image = imread(image_path)
        pipeline.process(image, model)
        if len(visual_images) < visual_count:
            visual_images.append(image)

    normalizer = Normalize()
    normalizer.normalize_data(model)
    normalizer.visualize(visual_images, pipeline.output_path("Normalize"), model.features_raw, model.features_scaled)

    x_train, x_test, y_train, y_test = train_test_split(model.features_scaled, model.labels, test_size=0.2)

    clf = Classify()
    clf.classify(x_train, y_train)
    acc = clf.accuracy(x_test, y_test)
    print("Accuracy is: ", acc)

if __name__ == "__main__":
    run_test_pipeline()
    pass
