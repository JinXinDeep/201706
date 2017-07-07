


"""
Generate training and test images.

"""
from numpy.core.setup_common import fname2def


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



OUTPUT_SHAPE = (64, 128)





def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M
    return M




def make_affine_transform(from_shape, to_shape,
                          rotation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = 1.0 

    size_tag = True
    while size_tag:
      roll = random.uniform(-1.6, 1.6) * rotation_variation
      pitch = random.uniform(-1.45, 1.45) * rotation_variation
      yaw = random.uniform(-1.45, 1.45) * rotation_variation
  
      # Compute a bounding box on the skewed input image (`from_shape`).
      M = euler_to_mat(yaw, pitch, roll)[:2, :2]
      h = from_shape[0]
      w = from_shape[1]
      #h, w = from_shape
      corners = numpy.matrix([[-w, +w, +w, -w],
                              [-h, -h, +h, +h]]) * 0.5
      
                       
      skewed_corners = M * corners
      skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                                numpy.min(M * corners, axis=1))
      if min(to_size - skewed_size) >0:
        size_tag = False

    

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    #scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) 
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])
    
    new_center = center_to + trans
    skewed_corners = skewed_corners + new_center
    
    left_cornor = new_center - skewed_size /2.
    right_cornor = new_center + skewed_size /2.
    print("right_cornor")
    print(right_cornor)
    print("left_cornor")
    print(left_cornor)

    return M, out_of_bounds, left_cornor, right_cornor, skewed_corners,skewed_size





def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out



def generate_bg(mydir):    
    classes = os.listdir(mydir)    
    nrof_classes = len(classes)
    tmpidx=random.randint(0, nrof_classes - 1)
    fname = os.path.join(mydir,classes[tmpidx])
    bg = cv2.imread(fname) / 255.
    return bg

def generate_plate(mydir):
    classes = os.listdir(mydir)    
    nrof_classes = len(classes)
    tmpidx=random.randint(0, nrof_classes - 1)
    fname = os.path.join(mydir,classes[tmpidx])
    my_image = cv2.imread(fname) / 255.

    return my_image, numpy.ones(my_image.shape)
  

def generate_im(id_img_dir, bg_img_dir):
    bg = generate_bg(bg_img_dir)
    tmps = (bg.shape[0], bg.shape[1])    
    while tmps[0] > 1500 and tmps[1] > 1500:
        tmps = (tmps[0] * 0.7, tmps[1] * 0.7)
    bg= cv2.resize(bg, (int(round(tmps[1])), int(round(tmps[0]))))

    plate, plate_mask = generate_plate(id_img_dir)
    
    # resize the ID image
    if random.uniform(0.0,1.0) <0.5:
        scale = random.uniform(0.3, 1)
    else:
        scale = random.randint(1,5)
        
    
    tmps = (plate.shape[0]*scale, plate.shape[1]*scale)
    
    while tmps[0] > bg.shape[0] or tmps[1] > bg.shape[1]:
        tmps = (tmps[0] * 0.7, tmps[1] * 0.7)

    
    plate = cv2.resize(plate, (int(round(tmps[1])), int(round(tmps[0]))))
    plate_before = plate
    plate_mask = numpy.ones(plate.shape)

    
    
    M, out_of_bounds, left_cornor, right_cornor, skewed_corners,skewed_size = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            rotation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    ##out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    
    trans_w = int(0.5*skewed_size[0])
    trans_h = int(0.5*skewed_size[1])
    left_cornor = left_cornor.astype(int)
    right_cornor = right_cornor.astype(int)
    
    left_trans_w =int( min(trans_w,left_cornor[0]))
    right_trans_w = int(min(trans_w,out.shape[1]-right_cornor[0]))
    top_trans_h = int( min(trans_h,left_cornor[1]) )
    bot_trans_h = int( min(trans_h,out.shape[0]-right_cornor[1]) )
    
    print(left_trans_w)
    print right_trans_w 
    print top_trans_h
    print bot_trans_h
    if left_trans_w >0:
      left_trans_w = random.randint(0, left_trans_w - 1)
    if right_trans_w >0:
      right_trans_w = random.randint(0, right_trans_w - 1)    
    if top_trans_h >0:
      top_trans_h = random.randint(0, top_trans_h - 1)
    if bot_trans_h >0:
      bot_trans_h = random.randint(0, bot_trans_h - 1)

   
    out = out[left_cornor[1,0]-top_trans_h:right_cornor[1,0]+bot_trans_h, left_cornor[0,0]-left_trans_w:right_cornor[0,0]+right_trans_w]
    
    margin_h = left_cornor[1,0]-top_trans_h
    margin_w = left_cornor[0,0]-left_trans_w
    
    skewed_corners = skewed_corners - numpy.matrix([[margin_w],
                            [margin_h]])

    return out, skewed_corners, not out_of_bounds, left_cornor, right_cornor, plate_before, plate



def generate_ims():
    """
    Generate composite images.

    :return:
        Iterable of  composite images.

    """
    variation = 1.0
    bg_img_dir=""
    id_img_dir=""
    num_bg_images = len(os.listdir(bg_img_dir)) 
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)


if __name__ == "__main__":
    
    out_dir="test"
    os.mkdir(out_dir)
    variation = 1.0
    bg_img_dir="G:/picture/bg"
    id_img_dir="G:/picture/id"
    id_img_back_dir="G:/picture/id_back"
    tag_file = "corner.txt"
    f = open(tag_file,'w') 
    #bg_img_dir="G:/picture/QQ"
    #id_img_dir="G:/picture/fj"
    num_out_image = 100
    for idx in range(num_out_image):
      front = True;         
      out_img, skewed_corners, flag, left_cornor, right_cornor, plate_before, plate = generate_im(id_img_dir, bg_img_dir)

      
      '''
      fname = "{:08d}_{}_plate_before.png".format(idx, "1" if flag else "0")
      fname = os.path.join(out_dir,fname)
      cv2.imwrite(fname, plate_before * 255.)
      
      fname = "{:08d}_{}_plate.png".format(idx, "1" if flag else "0")
      fname = os.path.join(out_dir,fname)
      cv2.imwrite(fname, plate * 255.)
      '''
      fname1 = "{:08d}_{}.jpg".format(idx, "1" if front else "0")
      fname = os.path.join(out_dir,fname1)
      cv2.imwrite(fname, out_img * 255.)
      skewed_corners = skewed_corners.astype(int)
      mytag = fname1
      for i in range(4):
        mytag = mytag + " " + str(skewed_corners[0,i])+ " " +str(skewed_corners[1,i])
      mytag = mytag +"\n"
      f.write(mytag)
      
    f.close()
                                 

        
    


