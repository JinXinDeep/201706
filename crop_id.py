
"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from xml.dom import minidom




if __name__ == "__main__":
    print("start")
    img_dir="G:/picture/all"
    id_img_dir="G:/picture/id"
    id_img_back_dir="G:/picture/id_back"
    
    classes = os.listdir(img_dir)    
    nrof_classes = len(classes)
    print(nrof_classes)
    for idx in range(nrof_classes):
      fname = classes[idx]
      if(6 == fname.rfind("1")):
        fname = os.path.join(id_img_dir,fname)
        img = cv2.imread(os.path.join(img_dir,classes[idx]))
        img = img[82:336,55:460]
        cv2.imwrite(fname, img)
      if(6 == fname.rfind("2")):
        fname = os.path.join(id_img_back_dir,classes[idx])
        img = cv2.imread(os.path.join(img_dir,classes[idx]))
        img = img[9:299,17:477]
        cv2.imwrite(fname, img) 
      
    

   
