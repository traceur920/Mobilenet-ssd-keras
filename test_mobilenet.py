#!/usr/bin/python3
import sys
sys.path.append("./models")
sys.path.append("./misc")
from mobilenet_v1 import mobilenet
m= mobilenet(input_tensor=None)

import tensorflow as tf
lite=tf.lite.TFLiteConverter.from_keras_model(m)
lite.summary()

