import tensorflow as tf
from tensorflow import keras

# Configuration for the architecture of the YOLOv1 model
architecture_config = [
    # Tuple format: (kernel_size, filters, stride, padding)
    (7, 64, 2, 3),
    "M",  # MaxPooling
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List format: [config1, config2, number_of_repeats]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=1, stride=1, padding=0, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)
        # Convolutional layer
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size, strides=stride, 
                                  padding='same' if padding != 0 else 'valid', use_bias=False)
        # Leaky ReLU activation function
        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.leakyrelu(x)
        return x

class Yolov1(keras.Model):
    def __init__(self, split_size, num_boxes, num_classes, in_channels=3):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        # Create the convolutional layers based on the architecture configuration
        self.darknet = self._create_conv_layers(architecture_config)
        # Create the fully connected layers
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def call(self, x, training=False):
        x = self.darknet(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.fcs(x, training=training)
        return x

    def _create_conv_layers(self, architecture):
        layers_list = []
        in_channels = self.in_channels

        # Loop through the architecture configuration to build the layers
        for x in architecture:
            if type(x) == tuple:
                layers_list.append(CNNBlock(x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]
            elif type(x) == str:  # MaxPooling layer
                layers_list.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            elif type(x) == list:  # Repeat the given layers
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers_list.append(CNNBlock(conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers_list.append(CNNBlock(conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]

        return keras.Sequential(layers_list)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # Fully connected layers for the model output
        return keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(496),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(S * S * (C + B * 5)),
        ])