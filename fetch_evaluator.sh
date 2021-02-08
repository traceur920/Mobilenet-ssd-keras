#! /bin/bash
git clone https://github.com/pierluigiferrari/ssd_keras.git
cp ssd_keras/data_generator/ -r ./
cp ssd_keras/bounding_box_utils/ -r ./
cp ssd_keras/ssd_encoder_decoder/ -r ./
cp ssd_keras/eval_utils/ -r ./
rm -r ssd_keras

patch eval_utils/average_precisions_evaluator.py < gpu.patch

#TODO: eventually implement this leveraging the new clone filtering feature from git 2.19+
