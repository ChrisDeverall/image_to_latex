from math import floor,ceil
import cv2
import numpy as np 
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from keras.models import load_model
import json

import sys

image_directory = sys.argv[1]
formula_count = int(sys.argv[2])
out_json_directory = sys.argv[3]

all_image_boxes=[]
model = load_model("symbol_detection/symbol_network.h5")
symbols = np.array(["a","b","c","x","y","z","+","-","*","(",")","0","1","2","3","4","5","6","7","8","9"])

for i in range(formula_count):
    filepath = image_directory + "/equ" + str(i) + ".jpg"
    img = cv2.imread(filepath,0)
    _,thresh = cv2.threshold(img,200,255, cv2.THRESH_BINARY_INV)

    cleared_thresh = clear_border(thresh)
    labeled = label(cleared_thresh)
    image_boxes = []

    for j in regionprops(labeled):
        if j.area > 8:
            y_min, x_min, y_max, x_max = j.bbox
            my_roi = img[y_min:y_max, x_min:x_max]
            formatted_image = cv2.resize(my_roi, (32,32), interpolation = cv2.INTER_AREA)
            formatted_image = np.expand_dims(formatted_image, axis=[0,3])
            predictions = model.predict(formatted_image)
            symbol_pred = symbols[np.argmax(predictions,axis=1)][0]
            box_dict = {"x" :x_min ,"y" : y_min,"w" : x_max - x_min,"h" : y_max - y_min,"label" : symbol_pred}
            image_boxes.append(box_dict)
    all_image_boxes.append(image_boxes)
    if i%1000 ==0:
        print(i)

with open(out_json_directory, 'w') as out_file:
    json.dump(all_image_boxes, out_file)