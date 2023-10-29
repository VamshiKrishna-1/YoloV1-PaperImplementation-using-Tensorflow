import os
import tensorflow as tf
import numpy as np
import pandas as pd
from dataset import VOCDataGenerator
from model import Yolov1
from loss import YoloLoss
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from utils import (cellboxes_to_boxes, non_max_suppression, plot_image)

# Set random seed for reproducibility
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)

# Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
EPOCHS = 80
CSV_FILE = "../Data/YoloV1-PaperImplementation-using-Tensorflow/1example.csv"
IMG_DIR = "../Data/YoloV1-PaperImplementation-using-Tensorflow/images"
LABEL_DIR = "../Data/YoloV1-PaperImplementation-using-Tensorflow/labels"
S = 7
B = 2
C = 20

# Model
model = Yolov1(split_size=S, num_boxes=B, num_classes=C)  # Make sure this is implemented in TensorFlow
model.build((None, 448, 448, 3))

# Loss and Optimizer
loss_fn = YoloLoss()  # Make sure this is implemented in TensorFlow
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def train_data_gen():
    data_generator = VOCDataGenerator(csv_file = CSV_FILE, img_dir = IMG_DIR, label_dir = LABEL_DIR, 
                                      S=7, B=2, C=20, transform=None, batch_size=BATCH_SIZE, shuffle=True)
    for data in data_generator:
        yield data

        
train_dataset = tf.data.Dataset.from_generator(
    train_data_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 448, 448, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 7, 7, 25), dtype=tf.float32)
    )
)



@tf.function
def train_step(imgs, labels):
    with tf.GradientTape() as tape:
        predictions = model(imgs, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train():
    best_loss = float('inf')
    num_samples = len(pd.read_csv(CSV_FILE))
    total_steps = -(-num_samples // BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        
        for imgs, labels in tqdm(train_dataset, total=total_steps):
            loss = train_step(imgs, labels)
            total_loss += loss.numpy()
            num_batches += 1
            print(f"\rEpoch {epoch + 1}/{EPOCHS}, Step {num_batches}, Loss: {loss}", end='')

        epoch_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    return model