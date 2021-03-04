#!/usr/bin/python3
import sys
sys.path.append("./models")
sys.path.append("./misc")
from ssd_mobilenet import ssd_300
scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]
#model=ssd_300('training', (300,300,3),20,scales=scales)
#model.summary()
img_height = 300
img_width = 300
n_classes = 20
model_mode = 'inference'
# 1: Build the Keras model
img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
subtract_mean = None  # The per-channel mean of the images in the dataset
#subtract_mean = [127.5,127.5,127.5]
swap_channels = [2,1,0]# The color channel order in the original SSD is BGR
swap_channels = False  # The color channel order in the original SSD is BGR
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
                      1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
                       1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]
aspect_ratios = [[1.001, 2.0, 0.5],[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [16, 32, 64, 100, 150, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True
conf_threshold = 0.5
model = ssd_300(model_mode,
    image_size=(img_height, img_width, img_channels),
    n_classes=n_classes,
    l2_regularization=0.0005,
    scales=scales,
    aspect_ratios_per_layer=aspect_ratios,
    two_boxes_for_ar1=two_boxes_for_ar1,
    steps=None,
    offsets=offsets,
    limit_boxes=limit_boxes,
    variances=variances,
    coords=coords,
    normalize_coords=normalize_coords,
    subtract_mean=subtract_mean,
    divide_by_stddev=127.5,
    swap_channels=swap_channels)
model.summary()
model.save("saved_model/model_for_v1_compat")
import tensorflow as tf
print("TENSORFLOW VERSION: ",tf.__version__)
converter=tf.compat.v1.lite.TFLiteConverter.from_saved_model("saved_model/model_for_v1_compat")
#converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.allow_custom_ops=True
lite=converter.convert()
with open('./model_v1_compat.tflite', 'wb') as f:
    f.write(lite)
