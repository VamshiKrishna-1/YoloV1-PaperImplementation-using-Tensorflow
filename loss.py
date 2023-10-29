import tensorflow as tf
from utils import intersection_over_union

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss, self).__init__()
        # Initialize Mean Squared Error loss
        self.mse = tf.keras.losses.MeanSquaredError(reduction = tf.losses.Reduction.SUM)
        # Grid size
        self.S = S
        # Number of bounding boxes
        self.B = B
        # Number of classes
        self.C = C
        # Loss weights
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def call(self, target, predictions):
        # Reshape the predictions to the desired shape
        predictions = tf.reshape(predictions, (-1, self.S, self.S, self.C + self.B*5))
        
        # Calculate Intersection over Union for both bounding boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        # Find the best bounding box based on IOU
        ious = tf.concat([tf.expand_dims(iou_b1, axis=0), tf.expand_dims(iou_b2, axis=0)], axis=0)
        best_box = tf.cast(tf.argmax(ious, axis=0), dtype=tf.float32)
        
        # Indicator for the presence of an object in a grid cell
        exists_box = target[..., 20:21]

        # Extract the predicted bounding box that has the highest IOU
        box_predictions = exists_box * (
            (1 - best_box) * predictions[..., 21:25] + best_box * predictions[..., 26:30]
        )

        box_targets = exists_box * target[..., 21:25]

        # Transform width and height values for better stability during training
        box_predictions = tf.concat([box_predictions[..., :2], 
                                     tf.sign(box_predictions[..., 2:4]) * tf.sqrt(tf.abs(box_predictions[..., 2:4]) + 1e-6), 
                                     box_predictions[..., 4:]], axis=-1)

        box_targets = tf.concat([box_targets[..., :2], 
                                 tf.sqrt(box_targets[..., 2:4]), 
                                 box_targets[..., 4:]], axis=-1)

        # Calculate Mean Squared Error for bounding box coordinates
        box_loss = self.mse(tf.reshape(box_targets, (-1, 4)), tf.reshape(box_predictions, (-1, 4)))

        # Calculate object loss for the responsible bounding box
        pred_box = best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        object_loss = self.mse(tf.keras.backend.flatten(exists_box * target[..., 20:21]),
                               tf.keras.backend.flatten(exists_box * pred_box))

        # Calculate no object loss for both bounding boxes
        no_object_loss = self.mse(tf.reshape((1 - exists_box) * target[..., 20:21], (-1,)),
                                  tf.reshape((1 - exists_box) * predictions[..., 20:21], (-1,)))
        no_object_loss += self.mse(tf.reshape((1 - exists_box) * target[..., 20:21], (-1,)),
                                   tf.reshape((1 - exists_box) * predictions[..., 25:26], (-1,)))

        # Calculate class probability loss
        class_loss = self.mse(tf.reshape(exists_box * target[..., :20], (-1, 20)),
                              tf.reshape(exists_box * predictions[..., :20], (-1, 20)))

        # Combine all the component losses to get the final loss
        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)

        return loss
