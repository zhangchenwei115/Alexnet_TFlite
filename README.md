Many thanks to WuZhe https://github.com/WZMIAOMIAO
all my code is copied from this Github
# Introduction
This model is used to detect 5 types of flowers. The name of this model is called Alexnet. What I am doing is to convert this model into Tensorflow Lite so that can be used in Rasberry Pi.  This is really simple but since I am a beginner of TensorFlow lite and this is the first model that I have successfully converted.  

# Env
windows 10  
python 3.8.3  
tensorflow 2.3.0

# How to train and convert TF-Lite:
Since this model is built with Keras. it is very easy to convert it with only a few codes behind the train part:
```bash
    converter =tfliteTFLiteConverterfrom_keras_mode(model)
    tflite_model = converter.convert()
```
Run the train file to train.
Then we can see the tflite file is generated.

# How to Use

download the flower [dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)
and put it into data_set/flower_data folder and use splite_data.py to split dataset into train and val.

In tensorflowlitetest file. we need to load Lite file and read the image then get output
```bash
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(img, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
```
**Attention**， predict.py file is the orignal file with tensorflow to test the flower. and tensorflowlitetest is the lite version to test the flower which i add a few code as above shows.

