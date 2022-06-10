import skimage.io as io
from PIL import Image
import glob
import os

Indices = [1, 2, 6, 7, 11, 12, 14, 17, 24, 29, 32, 34, 36, 37, 48, 61, 71, 72, 83, 99, 100, 161, 162, 168, 213, 221, 233, 7+235, 9+235, 11+235, 13+235, 14+235, 16+235, 18+235,22+235, 25+235, 27+235, 28+235, 29+235, 32+235, 38+235, 39+235, 41+235, 44+235,45+235, 49+235, 50+235, 53+235, 55+235,  58+235,  73+235,  86+235,  97+235,  116+235,  117+235]


out_dir = '/home/luis/Desktop/Proyecto_Meibo/unet-master_code_meibo/data/membrane/Mglands/test/'
for cnt in range(55):
	i=cnt+1
	i=0
    #img = io.imread(os.path.join(out_dir,"%d.BMP"%cnt),as_gray = True)
	Image.open((os.path.join(out_dir,"Mask%d.tif"%Indices[i]))).resize((340,256)).save(os.path.join(out_dir, str(Indices[i]) + '.png'))
	print(Indices[i])
