
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as iosk
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
from skimage import exposure, util
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from skimage import img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean
import time
import glob
from PIL import Image, ImageOps
import tensorflow as tf
import timeit
import zipfile
import io
import os
from skimage import img_as_ubyte

"""DEFS"""

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

"VANILLA"
#train_data_dir = "/home/luis/Desktop/DeepLearning/unet-master_code_vanilla/data/membrane/train/image/"
#label_data_dir= "/home/luis/Desktop/DeepLearning/unet-master_code_vanilla/data/membrane/train/label/"
"LID"
#train_data_dir = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/train/image/"
#label_data_dir = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/train/label/"
"MEIBO"
train_data_dir = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/Mglands/image/"
label_data_dir = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/Mglands/label/"


num_image=201
target_size = (256,256)
flag_multi_class = False
as_gray = True

imgs_mat=np.zeros((num_image,256,256,1))
lbl_mat=np.zeros((num_image,256,256,1))

for i in range(num_image):
  img = iosk.imread(os.path.join(train_data_dir,"%d.png"%i),as_gray = as_gray)
  img = trans.resize(img,target_size)
  img = exposure.equalize_adapthist(np.matrix.squeeze(img), clip_limit=0.04)#CLAHE
  img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
  img = np.reshape(img,(1,)+img.shape)
  imgs_mat[i,:,:,:]=img

  lbl = iosk.imread(os.path.join(label_data_dir,"%d.png"%i),as_gray = as_gray)
  lbl = trans.resize(lbl,target_size)

  lbl[lbl>.5]=1
  lbl[lbl==.5]=1
  lbl[lbl<.5]=0

  if np.max(lbl)>1:
    lbl = lbl / 255
    print("255")
  lbl = np.reshape(lbl,lbl.shape+(1,)) if (not flag_multi_class) else lbl
  lbl = np.reshape(lbl,(1,)+lbl.shape)
  lbl_mat[i,:,:,:]=lbl


model=[]
model = unet()
callbacks0 = [tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,verbose=1,mode="auto",baseline=None),
                tf.keras.callbacks.ModelCheckpoint('unet_membrane_meibogland_300.hdf5', monitor='loss',verbose=0, save_best_only=True)]

model.fit(imgs_mat,lbl_mat,batch_size=2,epochs=300,callbacks=callbacks0)

res = model.predict(img)
ress  = np.matrix.squeeze(res)
norm.shape
norm = (ress - np.min(ress)) / (np.max(ress) - np.min(ress))
norm[norm>.5]=1
norm[norm<.51]=0
print(np.min(imgs_mat))
print(np.max(imgs_mat))
plt.imshow(np.matrix.squeeze(norm))

plt.show()
plt.plot(np.matrix.squeeze(ress),'.')
#########################################################3

test_path_m = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/test"
num_res = 92

imgrs_mat  = np.zeros([num_res,256,256,1])
imgrs_mat.shape
for i in range(num_res):
  imgr = iosk.imread(os.path.join(test_path_m,"%d.png"%i),as_gray = as_gray)
  imgr = trans.resize(imgr,target_size)
  imgr = exposure.equalize_adapthist(np.matrix.squeeze(imgr), clip_limit=0.04)#CLAHE
  imgr = np.reshape(imgr,imgr.shape+(1,)) if (not flag_multi_class) else img
  imgr = np.reshape(imgr,(1,)+imgr.shape)
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
