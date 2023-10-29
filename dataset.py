import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

class VOCDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None, batch_size=4, shuffle=False):
        # Read annotations from the CSV file
        self.annotations = pd.read_csv(csv_file)
        # Image and label directories
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # Configurations for YOLO grid
        self.S = S
        self.B = B
        self.C = C
        # Batch size for data loading
        self.batch_size = batch_size
        # Indexes for data shuffling
        self.indexes = np.arange(len(self.annotations))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.annotations) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of image and label paths for the batch
        batch_img_paths = [os.path.join(self.img_dir, self.annotations.iloc[k, 0]) for k in indexes]
        batch_label_paths = [os.path.join(self.label_dir, self.annotations.iloc[k, 1]) for k in indexes]

        # Generate data for the batch
        X, y = self.__data_generation(batch_img_paths, batch_label_paths)

        return X, y
    
    
    def on_epoch_end(self):
        'Shuffles indexes after each epoch if shuffle is set to True'
        if self.shuffle:
            np.random.shuffle(self.indexes)    

    def __data_generation(self, batch_img_paths, batch_label_paths):
        'Generates data containing batch_size samples'
        # Initialize empty arrays for images and labels
        X = np.empty((len(batch_img_paths), 448, 448, 3))
        Y = np.empty((len(batch_label_paths), self.S, self.S, self.C + 5))

        for batch_num, (img_path, label_path) in enumerate(zip(batch_img_paths, batch_label_paths)):
            # Import and preprocess image
            image = Image.open(img_path)
            image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)  # Convert PIL Image to TensorFlow tensor
            image = image/225.0
            image = tf.image.resize(image, (448, 448))  # Resize image
            
            # Initialize empty list for boxes
            boxes = []
            with open(label_path) as f:
                # Read bounding box data from the file and append to the boxes list
                for label in f.readlines():
                    class_label, x, y, width, height = [
                        float(x) if float(x) != int(float(x)) else int(x)
                        for x in label.replace("\n", "").split()
                    ]
                    boxes.append([class_label, x, y, width, height])
                    boxes.append([class_label, x, y, width, height])
            boxes = np.array(boxes, dtype=np.float32)

            # Initialize empty label matrix
            label_matrix = np.zeros((self.S, self.S, self.C + 5), dtype=np.float32)
            
            # Convert bounding box data to YOLO grid format
            for box in boxes:
                class_label, x, y, width, height = box.tolist()
                class_label = int(class_label)

                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i
                width_cell, height_cell = width * self.S, height * self.S

                # Ensure i and j are within grid bounds
                i = min(i, self.S - 1)
                j = min(j, self.S - 1)
                if label_matrix[i, j, 20] == 0:
                    label_matrix[i, j, 20] = 1
                    box_coordinates = np.array([x_cell, y_cell, width_cell, height_cell], dtype=np.float32)
                    label_matrix[i, j, 21:25] = box_coordinates
                    label_matrix[i, j, class_label] = 1
                    
            X[batch_num] = image
            Y[batch_num] = label_matrix
                    
        return X, Y
