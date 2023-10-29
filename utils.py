import tensorflow as tf
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def intersection_over_union(boxes_preds, boxes_labels, box_format = 'midpoint'):
    """
    bboxes1, bboxes2 have shape (-1, self.S, self.S, 4)
    box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns: tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] 
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]


    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    # tf.clip_by_value is used here as the equivalent of .clamp(0)
    #intersection = tf.clip_by_value(x2 - x1, 0, x2 - x1) * tf.clip_by_value(y2 - y1, 0, y2 - y1)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                tf.constant(chosen_box[2:]),
                tf.constant(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()




def convert_cellboxes(predictions, S = 7):
    # Reshaping prediction tensor of shape (7, 7, 30)
    x = predictions.reshape((7, 7, 30))

    # Initialize new tensor of shape (7, 7, 25)
    new = np.zeros((7, 7, 25))

    # Copy class probabilities
    new[..., :20] = x[..., :20]

    # Reshape objectness scores to shape (2, 7, 7, 1)
    confd1 = x[..., 20:21].reshape(1, 7, 7, 1)
    confd2 = x[..., 25:26].reshape(1, 7, 7, 1)
    concat = tf.concat([confd1, confd2], axis=0)

    # Find the index of the maximum objectness score along the first axis
    arg_max = tf.argmax(concat, axis=0)

    # Create a boolean mask for selecting bounding box attributes
    mask = arg_max == 0  # This will be True where the first bounding box has higher confidence, False otherwise

    # Use the mask to select bounding box attributes from x and assign them to new
    new[..., 20:25] = np.where(mask, x[..., 20:25], x[..., 25:30])
    
    # Convert new to a TensorFlow tensor
    new = tf.convert_to_tensor(new, dtype=tf.float32)

    # Calculate the cell indices
    indices = tf.range(S, dtype=tf.float32)
    
    # Repeat and reshape to create a grid of indices
    cell_indices_x = tf.tile(indices[tf.newaxis, :], [S, 1])
    cell_indices_y = tf.tile(indices[:, tf.newaxis], [1, S])

    # Convert bounding box coordinates
    x = (new[..., 21:22] + cell_indices_x[..., tf.newaxis]) / S
    y = (new[..., 22:23] + cell_indices_y[..., tf.newaxis]) / S
    w_h = new[..., 23:25] / S

    converted_preds = tf.concat((x, y, w_h), axis=-1)
    
    class_pred = tf.math.argmax(new[..., :20], axis=-1)
    class_pred = tf.cast(class_pred, tf.float32)
    class_pred = tf.expand_dims(class_pred, -1)
    
    return tf.concat([class_pred, new[..., 20:21], converted_preds], axis = -1)

def cellboxes_to_boxes(out, S=7):

    return convert_cellboxes(out).numpy().reshape(49,6).tolist()