from model import AlexNet_v1, AlexNet_v2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf 

im_height = 224
im_width = 224

# load image
img = Image.open("8-1Q015154440-50.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# scaling pixel value to (0-1)
img = np.array(img) / 255.

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))


# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


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


predict_class = np.argmax(output_data)

print(class_indict[str(predict_class)], output_data[0][predict_class])
plt.show()


# im_height = 224
# im_width = 224

# # load image
# img = Image.open("8-1Q015154440-50.jpg")
# # resize image to 224x224
# img = img.resize((im_width, im_height))
# plt.imshow(img)

# # scaling pixel value to (0-1)
# img = np.array(img) / 255.

# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img, 0))

# # read class_indict
# try:
#     json_file = open('./class_indices.json', 'r')
#     class_indict = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)


# model = AlexNet_v1(class_num=5)
# model.load_weights("./save_weights/myAlex.h5")
# result = np.squeeze(model.predict(img))
# predict_class = np.argmax(result)
# print(class_indict[str(predict_class)], result[predict_class])
# plt.show()
