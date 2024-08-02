# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:16:39 2023

@author: Youcef
"""


from pandas import read_csv, DataFrame
from PIL import Image

# Defining the path of input images
path_in = "C:\\Users\\Lenovo\\Desktop\\TPIA\\TP5\\sign_road_images\\"

# Importing the csv file
data_in = read_csv(path_in + "road_sign_dataset.csv")

# Creating a dictionary with headers of the dataframe
dict = {'Filename':[],
        'Sign':[]
       }
data_out = DataFrame(dict)

# Defining the size of the output image
size_image = 128

# Resizing images and storing in a specific folder
path_out = "C:\\Users\\Lenovo\\Desktop\\TPIA\\TP5\\sign_road_48\\"
file_name = data_in['Filename'].values
class_name = data_in['Sign'].values
for i in range(len(file_name)):
    print(file_name[i])
    image = Image.open(path_in + file_name[i])
    new_image = image.resize((size_image, size_image))
    newfilename = "size_" + str(size_image) + "_" + file_name[i]
    new_row = {'Filename': newfilename, 
               'Sign': class_name[i]
                   }
    data_out.loc[len(data_out)] = new_row
    new_image.save(path_out + newfilename)

# Saving the csv file with the size of the image   
data_out.to_csv(path_out +  str(size_image) + "_road_sign_dataset.csv", index=False)

