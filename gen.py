


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

    roll = random.uniform(-0.1, 0.1) * rotation_variation
    pitch = random.uniform(-0.1, 0.1) * rotation_variation
    yaw = random.uniform(-0.2, 0.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h = from_shape[0]
    w = from_shape[1]
    #h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    
                     
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    

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
    
    left_cornor = new_center - skewed_size /2.
    right_cornor = new_center + skewed_size /2.
    print("right_cornor")
    print(right_cornor)
    print("left_cornor")
    print(left_cornor)

    return M, out_of_bounds, left_cornor, right_cornor





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

    
    
    M, out_of_bounds, left_cornor, right_cornor = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            rotation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    ##out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, not out_of_bounds, left_cornor, right_cornor, plate_before, plate

def create_tag_xml(tag_dir, filename, front, left_cornor, right_cornor, img):
    
    xml=minidom.Document()
    
    root=xml.createElement('annotation')
    
        
    xml.appendChild(root)
    
    folder_node=xml.createElement('folder')    
      
    folder_text=xml.createTextNode('my_folder')
    folder_node.appendChild(folder_text)
    root.appendChild(folder_node)  
    
    
    filename_node=xml.createElement('filename')
    
    filename_text=xml.createTextNode(filename[0:filename.rfind('.')])
    filename_node.appendChild(filename_text)
    root.appendChild(filename_node)
    
    path_node=xml.createElement('path')
    root.appendChild(path_node)
    path_text=xml.createTextNode(filename)
    path_node.appendChild(path_text)
    
    tmp_source = xml.createElement("source")
    tmp_database = xml.createElement("database")
    tmp_database.appendChild(xml.createTextNode("Unknown"))
    tmp_source.appendChild(tmp_database)
    root.appendChild(tmp_source)

    
    
    size_node=xml.createElement('size')
    root.appendChild(size_node)
    tmp = xml.createElement('width')
    tmp.appendChild(xml.createTextNode(str(img.shape[1])))
    size_node.appendChild(tmp)
    
    tmp = xml.createElement('height')
    tmp.appendChild(xml.createTextNode(str(img.shape[0])))
    size_node.appendChild(tmp)
    
    tmp = xml.createElement('depth')
    tmp.appendChild(xml.createTextNode("3"))
    size_node.appendChild(tmp)
    
    tmp = xml.createElement("segmented")
    tmp.appendChild(xml.createTextNode("0")) 
    root.appendChild(tmp)                  

                          
    object_node=xml.createElement('object')
    root.appendChild(object_node)
    if front:
      cls_name = "id_front"
    else:
      cls_name = "id_back"
    tmp = xml.createElement('name')
    tmp.appendChild(xml.createTextNode(cls_name))
    object_node.appendChild(tmp)
    
    tmp = xml.createElement('pose')
    tmp.appendChild(xml.createTextNode("Unspecified"))
    object_node.appendChild(tmp)
    
    tmp = xml.createElement('truncated')
    tmp.appendChild(xml.createTextNode("0"))
    object_node.appendChild(tmp)
    
    tmp = xml.createElement('difficult')
    tmp.appendChild(xml.createTextNode("0"))
    object_node.appendChild(tmp)

    
    bndbox_node = xml.createElement("bndbox")
    object_node.appendChild(bndbox_node)
    
    tmp = xml.createElement('xmin')
    tmp.appendChild(xml.createTextNode(str(int(left_cornor[0,0]))))
    bndbox_node.appendChild(tmp)
    
    tmp = xml.createElement('ymin')
    tmp.appendChild(xml.createTextNode(str(int(left_cornor[1,0]))))
    bndbox_node.appendChild(tmp)
    
    tmp = xml.createElement('xmax')
    tmp.appendChild(xml.createTextNode(str(int(right_cornor[0,0]))))
    bndbox_node.appendChild(tmp)
    
    tmp = xml.createElement('ymax')
    tmp.appendChild(xml.createTextNode(str(int(right_cornor[1,0]))))
    bndbox_node.appendChild(tmp)
    
    fname = filename[0:filename.rfind('.')] +".xml"
    fname = os.path.join(tag_dir,fname)    
    f=open(fname,'w')
    f.write(xml.toprettyxml(encoding='utf-8'))
    f.close()



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
    tag_dir = "Tag"
    os.mkdir(tag_dir)
    #bg_img_dir="G:/picture/QQ"
    #id_img_dir="G:/picture/fj"
    num_out_image = 10
    for idx in range(num_out_image):
      if random.uniform(0.0,1.0) <0.5:
        front = True;         
        out_img, flag, left_cornor, right_cornor, plate_before, plate = generate_im(id_img_dir, bg_img_dir)
      else:
        front = False;         
        out_img, flag, left_cornor, right_cornor, plate_before, plate = generate_im(id_img_back_dir, bg_img_dir)
      
        
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
      create_tag_xml(tag_dir, fname1, front, left_cornor, right_cornor, out_img)
        
    


