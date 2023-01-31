'''
Script to convert images that are not 3 channles (RGB) into 3 channels
'''

from PIL import Image     
import os       
paths = ["./data/training_potholes_images/","./data/testing_potholes_images/", "./data/pothole_image_data/pothole/"] # Source Folder 
for path in paths:
    for file in os.listdir(path):      
            extension = file.split('.')[-1]
            if extension == 'jpg':
                fileLoc = path+file
                img = Image.open(fileLoc)
                if img.mode != 'RGB':
                    print(file+', '+img.mode)
                    new = Image.new("RGB", img.size, (255, 255, 255))
                    new.paste(img,None)
                    new.save(file, 'JPEG', quality=100)