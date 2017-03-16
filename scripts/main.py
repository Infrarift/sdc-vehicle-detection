import glob
from cv2 import imread

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from FeaturePipeline import FeaturePipeline
from controllers.classify import Classify
from controllers.normalize import Normalize
from layers.colorhistogram import ColorHistogram
from layers.combine import Combine
from layers.crop import Crop
from layers.detect import Detect
from layers.heatmap import Heatmap
from layers.hog import Hog
from layers.identify import Identify
from layers.spatialbin import SpatialBin
from layers.windows import Windows
from models.featuremodel import FeatureModel
from models.model import Model
from models.searchmodel import SearchModel
from models.statemodel import StateModel
from pipeline import Pipeline
from trainer import Trainer

import pickle

def load_training_data():
    training_path = "../training_data/"
    trainer = Trainer(training_path, random_state=99)
    trainer.load_shuffle_split_join("vehicles/", 1)
    trainer.load_shuffle_split_join("non-vehicles/", 0)
    print("Data loading complete: {0} Features"
          .format(len(trainer.features)))
    return trainer

def run_test_pipeline():

    pipeline = FeaturePipeline(name="Test Pipeline", output_dir="../output_images/")
    model = FeatureModel()
    trainer = load_training_data()

    visual_images = []
    visual_count = 25

    #trainer.train_features = shuffle(trainer.features, random_state=4)
    #trainer.train_labels = shuffle(trainer.labels, random_state=4)

    for i in range(0, len(trainer.features)):
        image_path = trainer.features[i]
        label = trainer.labels[i]
        model.add_label(label)
        image = imread(image_path)
        pipeline.process(image, model)
        if len(visual_images) < visual_count:
            visual_images.append(image)

    normalizer = Normalize()
    normalizer.fit_scaler(model)
    normalizer.normalize_data(model)
    normalizer.visualize(visual_images, pipeline.output_path("Normalize"), model.features_raw, model.features_scaled)


    x_train, x_test, y_train, y_test = train_test_split(model.features_scaled, model.labels, test_size=0.025)

    clf = Classify()
    clf.normalizer = normalizer
    clf.classify(x_train, y_train)
    acc = clf.accuracy(x_test, y_test)
    pickle.dump(clf, open("clf.p", "wb"))
    print("Accuracy is: ", acc)

def create_search_pipeline(name = "Search Pipeline"):
    pipeline = Pipeline(name=name, output_dir="../output_images/")
    pipeline.add_layer(Crop(top_crop_ratio=0.55, bot_crop_ratio=0.1, side_crop_ratio=0.0))
    pipeline.add_layer(Windows())
    pipeline.add_layer(Identify())
    pipeline.add_layer(Heatmap())
    pipeline.add_layer(Detect())
    return pipeline

def create_search_model():
    global model
    clf = pickle.load(open("clf.p", "rb"))
    model = SearchModel()
    model.clf = clf
    state_model = StateModel()
    model.state_model = state_model
    print(model.state_model)
    return model

def run_search_pipeline_on_test_images():
    pipeline = create_search_pipeline()
    model = create_search_model()
    images = glob.glob("../test_images/*.jpg")
    for image_path in images:
        image = cv2.imread(image_path)
        pipeline.process(image, model)

def run_movie_pipeline():
    global pipeline

    create_search_model()
    pipeline = create_search_pipeline("Movie Pipeline")
    pipeline.should_save_images = False
    process_movie("../project_video.mp4", "../output_videos/project_video.mp4")

def process_movie(input_path, output_path):
    clip = VideoFileClip(input_path)
    out_clip = clip.fl_image(process_video_image)  # NOTE: this function expects color images!!
    out_clip.write_videofile(output_path, audio=False)

def process_video_image(video_image):
    global pipeline

    pipeline.should_save_images = True
    pipeline.should_convert_to_BGR = True
    return pipeline.process(video_image, model)

if __name__ == "__main__":
    #run_test_pipeline()
    #run_search_pipeline_on_test_images()
    run_movie_pipeline()
    pass
