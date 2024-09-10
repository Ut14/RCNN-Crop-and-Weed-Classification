# Uninstall and reinstall opencv-contrib-python to ensure the required package is used
!pip uninstall opencv-python -y
!pip install opencv-contrib-python

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import json
import warnings
import pickle

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Define paths for model and image data
model_path = '../input/rcnn-training-part-1-finetuning/RCNN_crop_weed_classification_model.h5'
test_img_path = '../input/rcnn-data-preprocessing-part-2/Test/'
images_path = '../input/crop-and-weed-detection-data-with-bounding-boxes/agri_data/data/'
svm_model_path = '../input/rcnn-training-part-2-cnn-svm/svm_classifier.pkl'
images_name = [x for x in os.listdir(images_path) if x.endswith('.jpeg')]

# Load the trained RCNN model
model = tf.keras.models.load_model(model_path)

# Display the model summary
model.summary()

# Extract the model up to the second last fully connected (FC) layer
model_without_last_two_fc = tf.keras.models.Model(model.inputs, model.layers[-5].output)
model_without_last_two_fc.summary()

# Load the pre-trained SVM model for classification
with open(svm_model_path, 'rb') as svm:
    svm_model = pickle.load(svm)

# Define a function to calculate the Intersection over Union (IoU)
def iou_calc(bb1, bb2):
    true_xmin, true_ymin, true_width, true_height = bb1
    bb_xmin, bb_ymin, bb_width, bb_height = bb2

    true_xmax = true_xmin + true_width
    true_ymax = true_ymin + true_height
    bb_xmax = bb_xmin + bb_width
    bb_ymax = bb_ymin + bb_height

    # Calculate areas
    true_area = true_width * true_height
    bb_area = bb_width * bb_height

    # Calculate intersection coordinates
    inter_xmin = max(true_xmin, bb_xmin)
    inter_ymin = max(true_ymin, bb_ymin)
    inter_xmax = min(true_xmax, bb_xmax)
    inter_ymax = min(true_ymax, bb_ymax)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    iou = inter_area / (true_area + bb_area - inter_area)

    # Ensure the IoU is within the valid range [0, 1]
    assert 0 <= iou <= 1
    return iou

# Define the detection function
def detection(img_path, confidence=0.9, iou_thresh=0.1):
    img = plt.imread(img_path)
    cv2.setUseOptimized(True)
    
    # Apply selective search to identify region proposals
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:2000]  # Get top 2000 region proposals
    
    pred_crop, pred_weed, final = [], [], []

    for rect in tqdm(rects):
        x, y, w, h = rect
        roi = img[y:y+h, x:x+w, :]
        resized_roi = cv2.resize(roi, (224, 224)) / 255.0
        
        # Extract features using the modified model
        feature = model_without_last_two_fc.predict(resized_roi.reshape(-1, 224, 224, 3))
        
        # Perform SVM classification
        pred_prob = svm_model.predict_proba(feature.reshape(-1, 4096))
        pred_label = svm_model.predict(feature.reshape(-1, 4096))

        if pred_label == 'crop' and np.max(pred_prob) > confidence:
            pred_crop.append([list(rect), np.max(pred_prob)])
        elif pred_label == 'weed' and np.max(pred_prob) > confidence:
            pred_weed.append([list(rect), np.max(pred_prob)])

    # Perform Non-Maximum Suppression (NMS) for both crop and weed classes
    def non_max_suppression(pred_boxes, label):
        if len(pred_boxes) == 0:
            return

        pred_score = [x[1] for x in pred_boxes]
        pred_bb = [x[0] for x in pred_boxes]

        for i in range(len(pred_boxes)):
            temp_bb, temp_score = pred_bb.copy(), pred_score.copy()
            if len(temp_bb) != 0:
                max_score_box = temp_bb[np.argmax(temp_score)]

                if [max_score_box, np.max(temp_score), label] not in final:
                    final.append([max_score_box, np.max(temp_score), label])

                    indices_to_remove = [ind for ind, other_bb in enumerate(temp_bb)
                                         if iou_calc(max_score_box, other_bb) >= iou_thresh]

                    pred_bb = [bb for idx, bb in enumerate(temp_bb) if idx not in indices_to_remove]
                    pred_score = [score for idx, score in enumerate(temp_score) if idx not in indices_to_remove]
                else:
                    continue
            else:
                break

    # Apply NMS to crop and weed predictions
    non_max_suppression(pred_crop, 'crop')
    non_max_suppression(pred_weed, 'weed')

    # Draw bounding boxes and save the result image
    imOut = img.copy()
    for rect, score, cls in final:
        x, y, w, h = rect
        color = (0, 255, 0) if cls == 'crop' else (255, 0, 0)
        cv2.rectangle(imOut, (x, y), (x + w, y + h), color, 2)
        cv2.putText(imOut, f'{cls}: {round(score * 100, 2)}%', (x, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    plt.imshow(imOut)
    cv2.imwrite('prediction.jpeg', imOut)

    return final

# Example usage
detection('../input/sampledataweed/data/1 (1).jpeg')
detection(images_path + images_name[24])
