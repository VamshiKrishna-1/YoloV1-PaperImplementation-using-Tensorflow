from train import train
import tensorflow as tf

# Training the model one example for demonstrating purposes
model = train()

from utils import (cellboxes_to_boxes, non_max_suppression, plot_image)
from PIL import Image
import numpy as np

# Importing and preprocessing the image
image = "../Data/YoloV1-PaperImplementation-using-Tensorflow/images/000007.jpg"

img = Image.open(image)
img = np.array(img)
img = img/225.0

# Convert the NumPy array to a TensorFlow tensor
tensor = tf.convert_to_tensor(img, dtype=tf.float32)
# Resize the image
resized_tensor = tf.image.resize(tensor, [448, 448])
resized_tensor = tf.expand_dims(resized_tensor, axis=0)
squeezed_tensor = tf.squeeze(resized_tensor, axis=0)
pred = model.predict(resized_tensor)
boxes = cellboxes_to_boxes(pred)
nms = non_max_suppression(boxes, iou_threshold = 0.8, threshold = 0.8)
plot_image(squeezed_tensor, nms)