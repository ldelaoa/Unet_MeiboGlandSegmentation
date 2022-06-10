from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.transform as trans
from skimage import exposure, util
from scipy import stats


as_gray = True
flag_multi_class = False
target_size = (256,256)

train_path_m = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/train/image/"
label_path_m = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/train/label/"
test_path = "/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/test"

train_path_v = "/home/luis/Desktop/DeepLearning/unet-master_code_vanilla/data/membrane/train/image/"
label_path_v = "/home/luis/Desktop/DeepLearning/unet-master_code_vanilla/data/membrane/train/label/"
test_path = "/home/luis/Desktop/DeepLearning/unet-master_code_vanilla/data/membrane/test"

i=6

#path = '/home/luis/Downloads/gris_lesion.jpeg'

img_label = io.imread(os.path.join(train_path),as_gray = as_gray)
img_label = trans.resize(img_label,target_size)

img_v = io.imread(os.path.join(label_path_v,"%d.png"%i),as_gray = as_gray)
img_m = io.imread(os.path.join(label_path_m,"%d.png"%i),as_gray = as_gray)
img_m = img / 255
img_m = trans.resize(img,target_size)
img_m = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
img_m = np.reshape(img,(1,)+img.shape)
#img_v = exposure.equalize_adapthist(np.matrix.squeeze(img), clip_limit=0.01)

img_v.dtype
img_v.shape
img_v.ndim
img_v.size
img_v.max()
img_v.min()

img_m.dtype
img_m.shape
img_m.ndim
img_m.size
img_m.max()
img_m.min()



[hist,bins] = exposure.histogram(img_label)
plt.plot(hist[4:255])
np.median(hist[4:255])
np.where(hist[4:255] == np.amax(hist[4:255]))


plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img)
plt.show()
