from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os
import pathlib
import train

#export the savedmodel
alexnet_saved_model = 'saved_model'
tf.saved_model.save(model, alexnet_saved_model)

#convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(alexnet_saved_model)
tflite_model = converter.convert()

#save the model
tflite_model_file = pathlib.Path('alexnet.tflite')
tflite_model_file.write_bytes(tflite_model)