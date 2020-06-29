
import numpy as np
import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import sys

image_directory = sys.argv[1]
images_per_label = int(sys.argv[2])

symbols = np.array(["a","b","c","x","y","z","+","-","*","(",")","0","1","2","3","4","5","6","7","8","9"])

images_train = np.empty([len(symbols)*images_per_label, 32, 32])
labels_train = np.empty([len(symbols)*images_per_label,len(symbols)])

counter = 0 
for i in symbols:
    for j in range(images_per_label):
        in_file_path = image_directory+"/"+str(i)+"/"+str(j)+".jpg"
        # in_file_path = "../data/symbols_train"+"/"+str("a")+"/"+str(1)+".jpg"

        img = cv2.imread(in_file_path,0)

        formatted_img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
        
        images_train[counter] = formatted_img
        labels_train[counter] = (symbols==i)
        counter +=1

perm = np.random.permutation(len(labels_train))
images_train = images_train[perm]
labels_train = labels_train[perm]

images_train = np.expand_dims(images_train, axis=3)
# ^ reformatting for CNN

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 1)))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(len(symbols), activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

model.fit(images_train, labels_train, batch_size = 64, epochs = 3, verbose = 1)

model.save("symbol_detection/symbol_network.h5")
#perform model.load to test / use network