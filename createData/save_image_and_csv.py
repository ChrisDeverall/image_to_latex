
import csv
from matplotlib import rc
import sys
import os

from createImages import *
from generateEquation import *



# from "createImages" import *
# from "generateEquations" import *

try:
    total_rows = int(sys.argv[1])
    image_directory = sys.argv[2]
    csv_name =  sys.argv[3]
except:
    print("Invalid Input!")
    print(str(sys.argv))


if os.path.isdir("./" + image_directory) ==False:
    os.mkdir(image_directory)

with open(csv_name, 'a') as file:
    writer = csv.writer(file)
    for i in range(0, total_rows):
        if i%1000 ==0:
            print(i)
        file_name = "./"+image_directory+"/equ"+str(i)+".jpg"
        complexity = i/total_rows
        my_formula, form_length = generateFormula(complexity)
        while form_length>34:
            my_formula, form_length = generateFormula(complexity)
        my_formula = "$$"+my_formula+"$$"
        my_row = [str(i),my_formula]
        writer.writerow(my_row)
        rc('font',**{'family':'serif','serif':['Bookman']}) 
        create_equation(my_formula, file_name, 30)