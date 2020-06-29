import keras
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model
import sys


image_directory = sys.argv[1]
images_per_label = int(sys.argv[2])

symbols = np.array(["a","b","c","x","y","z","+","-","*","(",")","0","1","2","3","4","5","6","7","8","9"])

images_test = np.empty([len(symbols)*images_per_label, 32, 32])
labels_test = np.empty([len(symbols)*images_per_label,len(symbols)])
counter = 0
for i in symbols:
    for j in range(images_per_label):
        in_file_path = image_directory+"/"+str(i)+"/"+str(j)+".jpg"

        img = cv2.imread(in_file_path,0)

        formatted_img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
        
        images_test[counter] = formatted_img
        labels_test[counter] = (symbols==i)
        counter += 1

perm = np.random.permutation(len(labels_test))
images_test = images_test[perm]
labels_test = labels_test[perm]

images_test = np.expand_dims(images_test, axis=3)

model = load_model("symbol_detection/symbol_network.h5")

score = model.evaluate(images_test, labels_test, verbose=0)
print("Test Accuracy:", score[1]*100, "%")