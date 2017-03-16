from layers.colorhistogram import ColorHistogram
from layers.combine import Combine
from layers.hog import Hog
from layers.spatialbin import SpatialBin
from pipeline import Pipeline

def FeaturePipeline(name = "Pipeline", output_dir = "../output_images", output_step = 1):
    pipeline = Pipeline(name=name, output_dir=output_dir, output_step=output_step)
    pipeline.dir_path = output_dir + "/" + name + "/"
    pipeline.output_step = output_step
    pipeline.add_layer(SpatialBin())
    pipeline.add_layer(ColorHistogram())
    pipeline.add_layer(Hog())
    pipeline.add_layer(Combine())
    return pipeline