import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from os import listdir
import time
import cv2
import json

from collections import defaultdict
from io import StringIO
from PIL import Image

sys.path.append("..")
from fmo_detection import detect_ball, plot_trajectory, from_json

import math

VIDEO_PATH = os.path.join("..", "train_data", "high_quality_videos", "batter")

# threshold for faster R-CNN: minimum output of the network to classify the object as a bat
SCORE_THRESH = 0.6
SHOW_IMAGES = False

# LOAD RELEVANT CODE AND MODEL FOR FASTER R-CNN

PATH_TENSORFLOW_API = os.path.join("models","research","object_detection")

MODEL_NAME = "rfcn_resnet101_coco_11_06_2017"
### AVAILABLE MODELS:
# fifth_best: 'ssd_mobilenet_v1_coco_11_06_2017'
# fourth_best: "rfcn_resnet101_coco_11_06_2017"
# third_best: "faster_rcnn_resnet101_coco_11_06_2017"
# second_best: "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017"
# best ging nicht: "faster_rcnn_nas_coco_24_10_2017"

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(PATH_TENSORFLOW_API, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TENSORFLOW_API, 'data', 'mscoco_label_map.pbtxt')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
NUM_CLASSES = 90

# This is needed since all models and utils code is stored in the object_detection folder of the tensorflow api
sys.path.append(os.path.join(PATH_TENSORFLOW_API, "utils"))
sys.path.append(os.path.join(PATH_TENSORFLOW_API))

import label_map_util

import visualization_utils as vis_util

if not os.path.exists(os.path.join(PATH_TENSORFLOW_API, MODEL_NAME)):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, PATH_TENSORFLOW_API)
print("succuessfully downloaded model")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Specify if faster R-CNN should only be applied on every x frame, and if only the swing should be evaluated
every_x_frame = 1
start_at_SWING_START = False
end_at_SWING_END = False
MIN_AREA=1000

# PROCESS ALL FILES

for files in os.listdir(VIDEO_PATH):
    INPUT_VIDEO_PATH = os.path.join(VIDEO_PATH, files)
    name = files[:-4]
    if files[0]==".":
        continue

    # FMO-C
    _, _, candidates_per_frame = detect_ball(INPUT_VIDEO_PATH, joints_array = None, plotting=False, min_area=MIN_AREA, every_x_frame=1) #400

    with open(os.path.join("outputs", name+ "_fmoc.json"), "w") as outfile:
        json.dump(candidates_per_frame, outfile)
    print("Finished FMO-C for file", files, "saved in outputs folder")

    # FASTER R-CNN

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if start_at_SWING_START:
      cap.set(cv2.CAP_PROP_POS_FRAMES, SWING_START)

    assert os.path.exists(INPUT_VIDEO_PATH), "INVALID INPUT VIDEO FILE"

    output_dic = {}
    batBox = []
    gloveBox = []

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
          # Definite input and output Tensors for detection_graph
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
          detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          if start_at_SWING_START:
              frame_count = SWING_START
          else:
              frame_count = 0
          if end_at_SWING_END:
              end = SWING_END
          else:
              end = np.inf
          while frame_count<end:
              ret, image_np = cap.read()
              if image_np is None:
                  break
              shape = image_np.shape

              # Only every x-th frame
              if (frame_count)%every_x_frame!=0:
                  continue

              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)

              # Actual detection.
              tic = time.time()
              (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

              print("Time for one image: ", time.time()-tic)

              # select the results where the object class is either bat or glove
              index = np.union1d(np.where(np.squeeze(classes)==40)[0],np.where(np.squeeze(classes)==39)[0]) #np.where(np.squeeze(classes)!=1)[0]
              classes = np.squeeze(classes)[index]
              scores = np.squeeze(scores)[index]
              boxes_normalized = np.squeeze(boxes)[index]

              # bounding boxes are outputted from the faster R-CNN in a normalized form, multiply with frame
              boxes = boxes_normalized * np.array([[shape[0],shape[1],shape[0], shape[1]] for _ in range(len(boxes))])
              num = len(classes)

              # save bounding boxes for bat and gloves
              inner_dic = {}
              for i, s in enumerate(scores):
                  if s>SCORE_THRESH:
                      if classes[i]==39:
                          class_name = "bat"
                          aabb = boxes[i].astype(int)
                          batBox.append([np.array([[aabb[1], aabb[0]], [aabb[3], aabb[2]]]), frame_count])
                      else:
                          aabb = boxes[i].astype(int)
                          gloveBox.append([np.array([[aabb[1], aabb[0]], [aabb[3], aabb[2]]]), frame_count])
                          class_name = "glove"
                      inner_dic[class_name] = {"score":float(s), "box":boxes[i].tolist()}
              output_dic[str(frame_count).zfill(4)] = inner_dic
              frame_count+=1
              # Visualization of the results of a detection.
              if SHOW_IMAGES:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes_normalized, classes.astype(np.int32), scores,
                    category_index,
                    use_normalized_coordinates=True,
                      min_score_thresh = SCORE_THRESH,
                      max_boxes_to_draw = 100,
                    line_thickness=8)
                  plt.figure(figsize=IMAGE_SIZE)
                  plt.imshow(image_np)
                  plt.title("Frame "+str(frame_count))
                  plt.show()

    batBox = np.array(batBox)
    oldBatBox = batBox
    gloveBox = np.array(gloveBox)

    print("FINISHED FASTER R-CNN for", name)

    # save faster R-CNN output
    with open(os.path.join("outputs", name+ "_fasterrcnn.json"), "w") as outfile:
      json.dump(output_dic, outfile)
