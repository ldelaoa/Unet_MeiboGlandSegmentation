
from __future__ import print_function
import skimage.io as iosk
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
from skimage import exposure, util
from skimage import img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean
import time
import glob
from PIL import Image, ImageOps
import timeit
import zipfile
import io
import os
from skimage import img_as_ubyte
from scipy import ndimage, misc
from sklearn import preprocessing
import cv2 as cv



#########################################################3
target_size = (256,256)

test_path_m = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/test"
num_res = 92

for i in range(num_res):
  imgr = iosk.imread(os.path.join(test_path_m,"%d.png"%i),as_gray = True)
  imgr = trans.resize(imgr,target_size)
  imgr1 = ndimage.prewitt(imgr)
  temp,imgr2 = cv.threshold(imgr1,127,255,cv.THRESH_BINARY)
  imgr3 = cv.dilate(imgr2,np.ones((5, 5), 'uint8'), iterations = 1)
  





  imgrs_mat[i,:,:,:]=imgr

resr = model.predict(imgrs_mat)

temp = np.round(np.matrix.squeeze(resr[1,:,:,:]))

save_path = "data/membrane/test"
for i in range(92):
    temp = np.matrix.squeeze(resr[i,:,:,:])
    norm = (ress - np.min(temp)) / (np.max(temp) - np.min(temp))
    norm[norm>.5]=1
    norm[norm<.51]=0
    iosk.imsave(os.path.join(save_path,"%d_predictv0_mgland.png"%i),img_as_ubyte(temp))

plt.imshow(temp)
