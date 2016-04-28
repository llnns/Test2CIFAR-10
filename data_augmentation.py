#based on https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/lib/data_augmentation.lua

def crop(x, offsets, width):
   height = width
   return x.crop(offsets[0], offsets[1], offsets[0] + width, offsets[1] + height)

def horizontal_reflection(x):
   return x.transpose(Image.FLIP_LEFT_RIGHT)
import numpy as np

def zoomout(x):
   return x.resize((24,24), Image.BILINEAR)

CROP_POS24 = []
def generate_crop_pos24():
   for i in range(0,8,4):
      for j in range(0,8,4):
        CROP_POS24.append((i,j))

CROP_POS28 = []
def generate_crop_pos28():
   for i in range(0,4,2):
      for j in range(0,4,2):
	     CROP_POS28.append((i,j))

generate_crop_pos24()
generate_crop_pos28()

def data_augmentation(x, y):
    scale = len(CROP_POS24) + len(CROP_POS28)
    new_x = np.zeros((len(x) * scale * 2, 3, 24, 24))
    new_y = np.zeros((len(x) * scale * 2))

    for i in range(len(x)):
        src = [i]
        images = []
        for j in range(len(CROP_POS24)):
            images.append(crop(src,CROP_POS24[j],24))

        for j in range(len(CROP_POS28)):
            images.append(zoomout(crop(src,CROP_POS28[j],28)))

        for j in range(len(images)):
            new_x[scale * 2 * (i - 1) + j] = images[j]
    	    new_y[scale * 2 * (i - 1) + j] = y[i]
    	    new_x[scale * 2 * (i - 1) + len(images) + j] = horizontal_reflection(images[j])
    	    new_y[scale * 2 * (i - 1) + len(images) + j] = y[i]

    return (new_x,new_y)
