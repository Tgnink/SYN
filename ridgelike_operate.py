#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Fjscah
@contact:***@**.com
@version: 1.0.0
@license: Apache Licence
@file: ridgelike_detector.py
@time: 2020/11/27 11:27
@description : $1
"""
from scipy.ndimage.filters import gaussian_filter
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian,threshold_local,threshold_yen
import matplotlib.pyplot as plt
import os
import os
import roifile
from roifile import ImagejRoi
import cv2
import PIL
import matplotlib
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import profile
from skimage.draw import polygon
import numpy as np
from TiffPseudoCapture import VideoCapture
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from tifffile import imwrite
def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image
def ridge_oprator(img):
    # return float64 array
    kwargs = {'sigmas': [1], 'mode': 'reflect', 'black_ridges': 0}
    img_local = threshold_local(img, 5)

    img_thre = img_local > threshold_yen(img_local)
    #img_local[~img_thre] = 0
    img_redge = frangi(img_local, **kwargs)
    #ridge_max = np.max(img_redge)
    #type_max = np.iinfo(img.dtype).max
    type_max = np.iinfo(np.uint8).max
    img_redge = np.array(img_redge * type_max,dtype=np.uint8)
    return img_redge

#image = color.rgb2gray(data.retina())[300:700, 700:900]
pathway="C:\\Users\\ZLY\\Desktop\\result\\morph"
roiname='RoiSet-0mg.zip'
roiname=os.path.join(pathway,roiname)

filelist=[
    "cell3-MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT-GPT+pep_CSU-561(2).tif",
    "MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT_CSU-561(2).tif",
    "MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT-GPT+PEP_CSU-561(2).tif"
]
#roi2[0].coordinates()
filename=os.path.join(pathway,filelist[0])

#filename='F:\\zly\\cell\\1.jpg'
img = Image.open(filename)

print(img.format, img.size, img.mode)
image = np.asarray(img)
#image=gaussian(image,3)
image = threshold_local(image, 5)
if len(image.shape)>2:
    image=color.rgb2gray(image)
height,width=image.shape
image = threshold_local(image, 5)
cmap = plt.cm.gray

kwargs = {'sigmas': [1], 'mode': 'reflect'}

fig, axes = plt.subplots(2, 5,sharex=True, sharey=True)
for i, black_ridges in enumerate([1, 0]):
    for j, func in enumerate([identity, meijering, sato, frangi, hessian]):
        kwargs['black_ridges'] = black_ridges
        result = func(image, **kwargs)



        axes[i, j].imshow(result, cmap=cmap, aspect='auto')
        if i == 0:
            axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                  'Sato\ntubeness', 'Frangi\nvesselness',
                                  'Hessian\nvesselness'][j])
        if j == 0:
            axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
imm=ridge_oprator(image)
imwrite('result.tif',imm)



