import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
from functools import reduce

import cv2
#print(cap)
#print(ret)
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/sai-kantareddy/Downloads/Aptiv/ssd_mobilenet_v2/frozen_inference_graph_uni.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/sai-kantareddy/Downloads/Aptiv/ssd_mobilenet_v2/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

hitf = open("result.csv",'w')
hitf.write('filename,width,height,class,bb0,bb1,bb2,bb3\n')
hitlim = 0.5
loop = 0
cap = cv2.VideoCapture('/home/sai-kantareddy/Downloads/videos/out.avi')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:

    while True:
      ret, image_np= cap.read()
      width,height,_=image_np.shape
      #image_np=cv2.imread('/home/sai-kantareddy/Downloads/Aptiv_object_detector/Input/b5a09604-69dd5d9f.jpg',1)
      #im_width,im_height,_=image_np.shape
      #print(image_np)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
      # nprehit = scores.shape[1] # 2nd array dimension
      # i=0

      loop+=1
      # localtime = time.asctime( time.localtime(time.time()) )
      # for j in range(nprehit):
      #     fname = 'frame_'+str(loop)+'_'+str(localtime)
      #     width=width
      #     height=height
      #     classid = int(classes[i][j])
      #     classname = category_index[classid]["name"]
      #     #t1=time.clock()
      #     score=scores[i][j]
      #     if (score>=hitlim):
      #         #sscore = str(score)
      #         bbox = boxes[i][j]
      #         bbox[0],bbox[1],bbox[2],bbox[3]=round(bbox[0]*height),round(bbox[1]*width),round(bbox[2]*height),round(bbox[3]*width)
      #         b0 = str(bbox[0])
      #         b1 = str(bbox[1])
      #         b2 = str(bbox[2])
      #         b3 = str(bbox[3])
      #         line = ",".join([fname,classname,str(width),str(height),b0,b1,b2,b3])
      #         hitf.write(line+"\n")
          #t2=time.clock()
          #print(round(t2-t1,3))
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      #hitf.flush()
      #hitf.close()
      #print(a,b)
      # new_boxes = []
      # for i, box in enumerate(np.squeeze(boxes)):
      #      if(np.squeeze(scores)[i] > 0.5):
      #         new_boxes.append([round(box[0]*im_height),round(box[1]*im_width),round(box[2]*im_height),round(box[3]*im_width)])
      #         np.savetxt('yourfile.csv', new_boxes, delimiter=',')
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      output_folder='/home/sai-kantareddy/Downloads/Aptiv/output'
      filename = 'frame_'+str(loop)
      output_path = (os.path.join(output_folder, filename+'.jpg'))
      cv2.imwrite(output_path, image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
