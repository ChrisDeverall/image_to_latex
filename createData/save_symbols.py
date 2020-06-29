from matplotlib import rc
import os
import sys
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
import cv2

from createImages import *
from generateEquation import *


image_directory = sys.argv[1]

images_per_label = int(sys.argv[2])


#need to create the folders 

symbols = np.array(["a","b","c","x","y","z","+","-","\\times","(", ")", "0","1","2","3","4","5","6","7","8","9"])


if os.path.isdir(image_directory+"/*") ==False: #assume if there's no multiplication there are no other folders
    for i in symbols:
        if i == "\\times":
            os.mkdir(image_directory+"/*")
        else:
            os.mkdir(image_directory+"/"+i)
is_fraction = False
dummy_image_path = "dummy_image.jpg"
for i in range(images_per_label):
    for j in symbols:
        rc('font',**{'family':'serif','serif':['Bookman']}) #adjusting font
        font_picker = random.random()
        if font_picker<0.25:
            rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
        elif font_picker<0.5:
            rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        elif font_picker<0.75:
            rc('font',**{'family':'serif','serif':['Palatino']})
        else:
            rc('font',**{'family':'serif','serif':['Bookman']})
        font_size = random.randint(28, 35)
        if j == "-" and random.random()>0.5: #this will make approximately 50% of minus signs fractions, so that they are all read as fractions
            is_fraction = True
            random_length_numerator = random.choice(["3","33","333"])
            fixed_size_denominator = "3"
            latex_expression = "\\frac{"+random_length_numerator+"}{"+fixed_size_denominator+"}"
            create_equation("$$"+latex_expression+"$$", dummy_image_path, font_size)
        else:
            create_equation("$$"+j+"$$", dummy_image_path, font_size)

        img = cv2.imread(dummy_image_path,0)
        if is_fraction == True:
            height = img.shape[0]
            halfway_height = int(height/2) - 6 #this was found to be the offset from halfway point
            img = img[halfway_height-6:halfway_height+6,:]
            is_fraction = False


        _,thresh = cv2.threshold(img,200,255, cv2.THRESH_BINARY_INV)
        cleared_thresh = clear_border(thresh)
        labeled = label(cleared_thresh)

        x_min,y_min,x_max,y_max = regionprops(labeled)[0].bbox
        my_roi = img[x_min:x_max, y_min:y_max]
        
        if j == "\\times":
            out_file_name = image_directory + "/*/"+str(i)+".jpg"
        else:
            out_file_name = image_directory + "/"+j+"/"+str(i)+".jpg"

        cv2.imwrite(filename = out_file_name,img= my_roi)

os.remove(dummy_image_path)


