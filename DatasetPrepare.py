# TEST READ JSON


import os
import sys
import random
import math
import re
import json
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import glob
import skimage.color
import skimage.io
import skimage.transform
import IPython
from PIL import Image
#path_to_json = "L:\\JAV Folder\\Test Frames\\CAWD-152\\Result\\frame145650.json"
path_to_dataset = "G:\\JAV Folder\\Test Frames"
path_to_output = 'G:\\jav folder\\OutputFolder'
def ReadJson (path_to_json):
    annotation = json.load(open(path_to_json))
    mask = np.zeros([annotation["imageHeight"], annotation["imageWidth"], 1], dtype=np.bool_)

    for i in range(len (annotation["shapes"])):
        if ((annotation["shapes"][i]["label"] == 'A') or (annotation["shapes"][i]["label"] == "Human")):
            pointsx,pointsy=zip(*annotation["shapes"][i]["points"])
            #rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (1920, 1080))
            rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (annotation["imageWidth"], annotation["imageHeight"]))
            mask[cc, rr, 0] = 1
            
    for i in range(len (annotation["shapes"])):
        if ((annotation["shapes"][i]["label"] != 'A') and (annotation["shapes"][i]["label"] != "Human")):
            pointsx,pointsy=zip(*annotation["shapes"][i]["points"])
            #rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (1920, 1080))
            rr, cc = skimage.draw.polygon(pointsx, pointsy, shape = (annotation["imageWidth"], annotation["imageHeight"]))
            mask[cc, rr, 0] = 0
    return mask

def LoadDataset (path_to_dataset):
    print ("Reading preloaded dataset")
    ImageList = np.load(path_to_dataset + "\\ImageList.npy")
    JsonList = np.load(path_to_dataset + "\\JsonList.npy")
    return ImageList, JsonList

def LoadDatasetCustom (path_to_dataset, ImageName = "ImageList_512_Binary.npy", JsonName = "JsonList_512_Binary.npy"):
    print ("Reading preloaded BINARY dataset")
    ImageList = np.load(path_to_dataset + "\\" + ImageName)
    JsonList = np.load(path_to_dataset + "\\" + JsonName)
    return ImageList, JsonList

def readAllImage (path_to_image, height=540, width=960):
    count = 0
    for Files in glob.glob(path_to_dataset + "\\*\\*.png"):
            count+=1
    ImageList = numpy.empty ((count, height, width, 3), dtype=np.uint8)
    count = 0
    for Files in glob.glob(path_to_dataset + "\\*\\*.png"):
            Image = cv2.imread(Files)
            #ImageList[count] = np.expand_dims(cv2.resize(Image, (height, width)),0)
            ImageList[count] = np.expand_dims(cv2.resize(Image, (width, height)),0)
            count+=1
    return ImageList

def DatasetPrepare (path_to_dataset, height, width, saveToFile = 1, is_binary = 0,Postfix = ""):
    count = 0
    bak_count = 0
    for Folder in glob.glob(path_to_dataset + "\\*\\"):
        bak_count = count
        for JsonFiles in glob.glob (Folder + "Result\\*"):
            count+=1
        #print (-bak_count + count, " " ,Folder)
        #print("")
    print ("Counted [DatasetPrepare]")
    print (count)
    #return 0,0
    if (is_binary == 0):
        ImageList = numpy.empty ((count, height, width, 3), dtype=np.uint8)
    else:
        ImageList = numpy.empty ((count, height, width, 1), dtype=np.uint8)
    JsonList = numpy.empty ((count, height, width, 1), dtype=np.bool_)
    FileNameList = []
    count = 0
    for Folder in glob.glob(path_to_dataset + "\\*\\"):
        for JsonFiles in glob.glob (Folder + "Result\\*"):
            try:
                JsonData = ReadJson (JsonFiles)
            except:
                print ("Error")
                print (JsonFiles)
                print (Folder)
                continue
            if (is_binary == 0):
                Image = cv2.resize(
                            cv2.imread(Folder + ((JsonFiles.split("\\"))[-1])[0:-5] + ".png")
                                   ,(width, height))
            else:
                Image =np.expand_dims(
                        cv2.resize(
                            cv2.imread(Folder + ((JsonFiles.split("\\"))[-1])[0:-5] + ".png",
                                        cv2.IMREAD_GRAYSCALE)
                        ,(width, height))
                       ,-1)
            FileNameList.append (JsonFiles.split("\\")[-3] + '_' + (JsonFiles.split("\\")[-1])[0:-5]) 
            ImageList[count] = np.expand_dims(Image,0)
            JsonList[count] = np.expand_dims(skimage.transform.resize(JsonData, (height, width),anti_aliasing= False),0)
            count+=1
    if (saveToFile == 1):
        np.save(path_to_dataset + "\\ImageList" + Postfix, ImageList)
        np.save(path_to_dataset + "\\JsonList" + Postfix, JsonList)
    return ImageList, JsonList, FileNameList
def SaveDataset (OutputPath, ImageList, JsonList, FileNameList):
    ImgPath = os.path.join(OutputPath, "image")
    MaskPath = os.path.join(OutputPath, "mask")
    if not os.path.exists(ImgPath):
        os.makedirs(ImgPath)
    if not os.path.exists(MaskPath):
        os.makedirs(MaskPath)
    
    for img_array, name in zip(ImageList, FileNameList):
        # Convert the NumPy array to an image
        img = Image.fromarray(img_array[..., ::-1])
        # Save the image
        img.save(os.path.join(ImgPath, name+'.png'))
        
    for mask_array, name in zip(JsonList, FileNameList):
        # Convert the NumPy array to an image
        mask = Image.fromarray(mask_array[:,:,0])
        # Save the image
        mask.save(os.path.join(MaskPath, name+'.png'))
    
if __name__ == "__main__":
    #print ("Running")
    #start = time.time()
    ImageList, JsonList, NameList = DatasetPrepare (path_to_dataset, 520, 924, saveToFile = 0, is_binary = 0,Postfix = "_512_Binary")
    SaveDataset (path_to_output,ImageList, JsonList, NameList)

    #ImageList, JsonList = DatasetPrepare (path_to_dataset, 512, 512,
    #                                      saveToFile = 1, is_binary = 1,Postfix = "_512_Binary")
