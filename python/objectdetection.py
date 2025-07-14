import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import cv2
import tensorflow as tf
import collections
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from matplotlib import patches
from matplotlib.patches import Rectangle

# Load the TensorFlow model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

# Load the label map
def load_label_map():
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

category_index = load_label_map()

# Load the model
def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = load_model()

# Run object detection
def run_inference_for_single_image(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Expand dimensions since the model expects images to be in the form of [1, None, None, 3]
            image = np.expand_dims(image, axis=0)

            # Actual detection.
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # All outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            # Handle the case where the number of detections is less than the maximum number of detections
            if 'detection_masks' in output_dict:
                output_dict['detection_masks_reframed'] = output_dict['detection_masks_reframed'][0]
                output_dict['detection_masks_reframed'] = (output_dict['detection_masks_reframed'] > 0.5).astype(np.uint8)
            return output_dict

# Visualization of the results
def visualize_results(image, output_dict):
    # Visualization of the results of a detection.
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for i in range(output_dict['num_detections']):
        if output_dict:
            
