t = time.time()

Indices = [1, 2, 6, 7, 11, 12, 14, 17, 24, 29, 32, 34, 36, 37, 48, 61, 71, 72, 83, 99, 100, 161, 162, 168, 213, 221, 233, 7+235, 9+235, 11+235, 13+235, 14+235, 16+235, 18+235,22+235, 25+235, 27+235, 28+235, 29+235, 32+235, 38+235, 39+235, 41+235, 44+235,45+235, 49+235, 50+235, 53+235, 55+235,  58+235,  73+235,  86+235,  97+235,  116+235,  117+235]

from keras.models import load_model
import numpy as np
import os
import skimage.io as iosk
import time
from skimage import exposure, util
import skimage.transform as trans
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from skimage import img_as_ubyte
import time

def Tanimoto(A,B):
#Tanimoto Coefficient
#The Tanimoto similarity is only applicable for a binary variable,  ...
#and for binary variables the Tanimoto coefficient ranges from 0 to +1 (where +1 is the highest similarity).
#The Tanimoto coefficient between two points, a and b, with k dimensions is calculated as:

#A y B  - datos
#k - tamaño

#   (A * B)
#______________
#(A² + B² - A*B)

 index = np.sum(np.dot(A,B)) / (np.sum(np.dot(A,A)) + np.sum(np.dot(B,B)) - np.sum(np.dot(A,B)));

 return index


model = load_model('/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/unet_membrane_lid_300.hdf5')
meibo_model = load_model('/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/unet_membrane_meibogland_300.hdf5')

test_path_m = '/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/train/image/'
out_dir = '/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/Mglands/test/'


tani = np.zeros(np.size(Indices))
tani2 = np.zeros(np.size(Indices))
bacc = np.zeros(np.size(Indices))
bacc2 = np.zeros(np.size(Indices))
elapsed = np.zeros(np.size(Indices))

for i in range(np.size(Indices)):
    imgr = iosk.imread(os.path.join(test_path_m,"%d.png"%Indices[i]),as_gray = True)
    imgr = trans.resize(imgr,(256,256))
    imgr = exposure.equalize_adapthist(np.matrix.squeeze(imgr), clip_limit=0.04)#CLAHE
    imgr = np.reshape(imgr,imgr.shape+(1,)) if (not False) else imgr
    imgr = np.reshape(imgr,(1,)+imgr.shape)
    res_lid = model.predict(imgr)
    res_lid = (res_lid - np.min(res_lid)) / (np.max(res_lid) - np.min(res_lid))
    res_lid[res_lid>.5]=1
    res_lid[res_lid<.51]=0

    lid_Index  = np.matrix.squeeze(res_lid)
    lid_Index = np.where(lid_Index == 1)
    [x,y] = (np.asarray(lid_Index))


    res_meibo = meibo_model.predict(imgr)
    res_meibo = (res_meibo - np.min(res_meibo)) / (np.max(res_meibo) - np.min(res_meibo))
    res_meibo[res_meibo>.5]=1
    res_meibo[res_meibo<.51]=0
    ress = ((np.matrix.squeeze(res_meibo)*-1)+1) * np.matrix.squeeze(res_lid)
    #iosk.imsave(os.path.join(out_dir,"%d_Res_mgland.png"%i),img_as_ubyte(ress))

    lbl = iosk.imread(os.path.join(out_dir,"%d.png"%Indices[i]),as_gray = True)
    lbl = trans.resize(lbl,(256,256))
    lbl[lbl>.5]=1
    lbl[lbl<.51]=0
    lbl = (lbl*-1)+1


    elapsed[i] = time.time() - t
    t = time.time()

    tani[i]=Tanimoto(ress[y,x],(lbl[y,x]))
    tani2[i]=Tanimoto(ress.ravel(),lbl.ravel())
    bacc[i] = balanced_accuracy_score(ress[y,x],(lbl[y,x]))
    bacc2[i]= balanced_accuracy_score(ress.ravel(),(lbl.ravel()))

    #plt.subplot(1,3,1),plt.imshow(np.matrix.squeeze(res_meibo)),plt.title("meibo seg")
    #plt.subplot(1,3,2),plt.imshow(ress),plt.title("meibo*lid")
    #plt.subplot(1,3,3),plt.imshow(lbl),plt.title("label")
    #plt.show()

print("bacc",np.mean(bacc[i]),np.std(bacc[i]))
print("bacc2",np.mean(bacc2[i]),np.std(bacc[i]))
print("tani1",np.mean(tani[i]),np.std(bacc[i]))
print("tani2",np.mean(tani2[i]),np.std(bacc[i]))
print("tiempo",np.mean(elapsed[i]),np.std(bacc[i]))

#plt.subplot(2,2,3),plt.imshow(ress)
#plt.subplot(2,2,4),plt.imshow(lbl*-1)
