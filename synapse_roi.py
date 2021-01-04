#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   labelroi.py
@Time    :   2020/11/27 19:18:01
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, LNP2 group
@Desc    :   label synapse dendrite and background
'''

# here put the import lib
# here put the import lib
from operator import mul
import os
from os import system
from matplotlib.pyplot import xcorr
from numpy.core import function_base
import roifile
from roifile import ImagejRoi
import cv2
import PIL
import matplotlib
from PIL import Image, ImageDraw
from tifffile import imread, imwrite
from matplotlib import pyplot as plt
import profile
from skimage.draw import polygon
import numpy as np
from skimage.morphology.max_tree import area_opening
from TiffPseudoCapture import VideoCapture
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import data
from skimage.feature import match_template
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future, color
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from skimage.filters import meijering, sato, frangi, hessian, frangi, threshold_local, try_all_threshold, threshold_yen,threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import diameter_closing, diameter_opening, opening, skeletonize, closing, area_closing, remove_small_objects, binary_opening,convex_hull_image
from skimage import data
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from skimage.morphology import closing
from skimage.morphology import square
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from fil_finder import FilFinder2D
from astropy.io import fits
import astropy.units as u
from skimage.color import label2rgb
from skimage import measure
import cv2
import random
import zipfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import func_base
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import shift_roi
DEFAULT_COLORS = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,255,0),
                  (255,0,255)]
#-----------------------#
#   0. function  #
#-----------------------#
def branch_sklenton2(img):
    #fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    # need little large, otherwise may too local so that a windown only has axon not background
    kwargs = {'sigmas': [1], 'mode': 'reflect', 'black_ridges': 0}
    img_local = threshold_local(img, 35)
    #img_local[~img_thre] = 0
    img_redge = frangi(img_local, **kwargs)
    #ridge_max = np.max(img_redge)
    #type_max = np.iinfo(img.dtype).max
    type_max = np.iinfo(np.uint8).max
    img_redge = np.array(img_redge * type_max,dtype=np.uint8)

    img_thre = img_redge > threshold_yen(img_redge)
    img_skeleton=skeletonize(img_thre)
    #branch_sklenton(imm)
    #ax[0].imshow(img_skeleton,cmap='gray')
    fil = FilFinder2D(img_skeleton, mask=img_skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=40 * u.pix, prune_criteria='length')
    plt.figure()
    plt.imshow(fil.skeleton)
    plt.show()
    # Show the longest path
    #ax[1].imshow(fil.skeleton, cmap='gray')
    #plt.contour(fil.skeleton_longpath, colors='r')
    #plt.axis('off')
    #plt.show()
    # print("filafila~~~~~",fil.filaments)
    # fil1 = fil.filaments[0]
    # fil1.skeleton_analysis(fil.image, verbose=True, branch_thresh=5 * u.pix, prune_criteria='length')
    # print("fill length",fil.lengths(),fil.branch_properties.keys())

    return fil.skeleton
def branch_sklenton(img):
    #fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    # need little large, otherwise may too local so that a windown only has axon not background
    img_local = threshold_local(img, 25) 
    # plt.figure()
    # plt.hist(img_local.ravel(),bins=1000)
    # plt.show()

    img_thre = img_local > threshold_yen(img_local)
    img_skeleton=skeletonize(img_thre)
    #branch_sklenton(imm)
    #ax[0].imshow(img_skeleton,cmap='gray')
    fil = FilFinder2D(img_skeleton, mask=img_skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=30* u.pix, skel_thresh=30 * u.pix, prune_criteria='length')

    
    # Show the longest path
    #ax[1].imshow(fil.skeleton, cmap='gray')
    #plt.contour(fil.skeleton_longpath, colors='r')
    #plt.axis('off')
    #plt.show()
    # print("filafila~~~~~",fil.filaments)
    # fil1 = fil.filaments[0]
    # fil1.skeleton_analysis(fil.image, verbose=True, branch_thresh=5 * u.pix, prune_criteria='length')
    # print("fill length",fil.lengths(),fil.branch_properties.keys())

    return fil.skeleton


def label_roi(img,roiname,labeledimg=None):
    # return lebeled image, fiber 2 synapse 3 background 1
    height, width = img.shape
    if labeledimg:
        # has labeled fiber
        mask=labeledimg.copy()
    else:
        # roughly fiber roi
        mask = np.zeros((height, width), dtype=np.uint8)
        # need little large, otherwise may too local so that a windown only has axon not background
        img_local = threshold_local(img, 35) 
        img_thre = img_local > threshold_yen(img_local)
        #imm=skeletonize(img_thre)
        #branch_sklenton(imm)
        mask[img_thre] = 2
    # load synapse roi
    roi2 = ImagejRoi.fromfile(roiname)
    # label synapse roi
    for roi in roi2:
        coord = roi.coordinates()  # np.array that (n,2)
        #print(coord.shape)
        xcord = coord[:, 0] - 1
        ycord = coord[:, 1] - 1

        rr, cc = polygon(xcord, ycord)
        mask[cc, rr] = 3
        mask[ycord, xcord] = 1
    return mask
    mask[:50, :50] = 1
    mask[250:300, 120:160] = 1
    #plt.imsave('mask.png',mask)
    ax[3].imshow(mask)
    ax[3].set_title('labeled(manual)')
    plt.show()


    pass

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

def synapse_roi_mask(img_ridge,img_labeled):
    # return binary bool aray
    img_ridge[img_labeled!=3]=0
    # binary
    thresh = threshold_yen(img_ridge)
    img_binary = img_ridge> thresh
    # fill hole
    img_binary = ndi.binary_fill_holes(img_binary)
    label_objects, nb_labels = ndi.label(img_binary)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes =np.bitwise_and(sizes > 6 , sizes<100)
    mask_sizes[0] = 0
    img_binary = mask_sizes[label_objects]
    return img_binary

def contour_filter(contours):
    #exlit too small < 10 and to thin(maybe fiber)

    contours_new=[]
    for contour in contours:
        hull = ConvexHull(contour,qhull_options='QG4')
        hull_coord=[[x[1],x[0]] for x in hull.vertices]
        contours_new.append(hull_coord)
    return contours_new
def synapse_contour(img_binary,elite=4):
    contours_new=[]
    img_binary=np.array(img_binary*255,dtype=np.uint8)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area=cv2.contourArea(hull)

        # min rect
        rect = cv2.minAreaRect(hull)
        # 矩形四个角点取整
        box = np.int0(cv2.boxPoints(rect))
        d1=np.linalg.norm(box[0]-box[1])
        d2=np.linalg.norm(box[1]-box[2])
        if area<8 or hull_area/area>2.5:
            continue
        elif d1/d2>elite or d2/d1>elite:
            continue
        else:
            contours_new.append(hull)

        
        #cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)
    return contours_new
def write_roi(contours,outfile,img):
    # outfile +zip # outfile +tif
    if os.path.exists(outfile+'_roi.zip'):
        os.remove(outfile+'_roi.zip')
    zip1 = zipfile.ZipFile(outfile+'_roi.zip', 'w')
    zip1.close()
    mask=np.zeros_like(img,dtype=np.uint8)
    mask=cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    for cnt in contours:
        cnt=np.squeeze(cnt)
        roi = ImagejRoi.frompoints(cnt)
        roi.tofile(outfile+'_roi.zip')
    imwrite(outfile+'_roi.tif',mask)
    # out img
    # draw roi


def analysis(template,targetlist,outlist,expand_pixel=5):
    #---load img---#
    print(template,targetlist,outlist)
    img = imread(template)
    # synapse roi
    img_label=label_roi(img,template)
    # label background
    img_label[:50, :50] = 1
    #img_label[250:300, 120:160] = 1
    #--- ridgelike : fiber and synapse---#expand_pixel=5
    img_ridge=ridge_oprator(img)
    #---feiber skeleton---#
    img_skeleton=branch_sklenton(img)
    #img_copy = binary_opening(img_copy,square(diameter))
    #img=remove_small_objects(img, 100, connectivity=2)
    print('ori type', img.dtype, 'max', np.max(img))
    print('thre dtype ', img_skeleton.dtype, 'max', np.max(img_skeleton))
    print('ridge dtype ', img_ridge.dtype, 'max', np.max(img_ridge))

    #img_redge.dtype=img.dtype
    #img_redge[~img_thre]=0

    # !!!!!
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 4))
    ax[0,0].imshow(img, cmap=cm.gray)
    ax[0,0].set_title('ori')  # only adaptive threshold
    ax[0,1].imshow(img_skeleton, cmap=cm.gray)
    ax[0,1].set_title('fiber')  # auto threshold
    ax[0,2].imshow(img_ridge, cmap=cm.gray)
    ax[0,2].set_title('fiber+synapse')  # detector ridge like region
    ax[1,0].imshow(img_label)
    ax[1,0].set_title('labeled(manual)')
    plt.show()
    # !!!!!
    # exit()

    #--- primary segment trainning : cannot identify roi exactly  ---#
    training_labels = img_label
    sigma_min = 1
    sigma_max = 5
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True,
                            edges=True,
                            texture=True,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            multichannel=False)
    features = features_func(img)
    clf = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        warm_start=True,
        class_weight=None,
        ccp_alpha=0,
        max_samples=0.05,
    )
    #---5. self test---#
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)

    img_ridge_t = img_ridge.copy()
    #cimg=hessian(img,**kwargs)
    print('result label', set(list(result.ravel())))
    img_ridge_t[result != 3] = 0
    img_binary=synapse_roi_mask(img_ridge_t,result)
    labeled_img, _ = ndi.label(img_binary)
    image_label_overlay = label2rgb(labeled_img, image=img, bg_label=0,image_alpha=1)
    synapse_contour(img_binary)
    contours = measure.find_contours(img_binary, 0.8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9, 4))
    ax[0].imshow(img_binary, cmap=plt.cm.gray)

    #contours=contour_filter(contours)
    contours=synapse_contour(img_binary)
    n=0
    for n, contour in enumerate(contours):
        # hull = ConvexHull(contour)
        # # print('***',contour)
        # # print(hull.points)
        # points=np.array(hull.points)
        # ax[0].plot(points[:,1],points[:,0],linewidth=2)
        ax[0].plot(contour[:,:, 0], contour[:,:,1], linewidth=2)
    print('> roi number : ',n)

    ax[1].imshow(img,cmap='gray')
    ax[2].imshow(img_label)
    plt.show()



    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 4))
    for cnt in contours:
        colo=random.sample(DEFAULT_COLORS, 1)[0]
        #print(colo)
        image_label_overlay=cv2.drawContours(image_label_overlay, [cnt], -1, colo, -1)
    #ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
    #ax[0].contour(training_labels)
    ax[0,0].imshow(img,cmap='gray')
    ax[0,0].set_title('ori')
    ax[0,1].set_title('ori mask')
    ax[0,1].imshow(img_label,cmap='gray')
    ax[0,2].imshow(result,cmap='gray')
    ax[0,2].set_title('Segmentation')
    ax[1,0].imshow(img_ridge_t,cmap='gray')
    ax[1,0].set_title('result mask')
    ax[1,1].imshow(img_binary,cmap='gray')
    ax[1,1].set_title('final mask')
    ax[1,2].imshow(image_label_overlay)
    ax[1,2].set_title('final label')
    fig.tight_layout()

    plt.show()

    




    #---7.predict---#
    for filename,outname in zip(targetlist,outlist):

        #img = Image.open(filename)
        img = Image.open(filename)
        img = np.asarray(img)
        img_label=label_roi(img,filename)
        #--- ridgelike : fiber and synapse---#
        img_ridge=ridge_oprator(img)
        #---feiber skeleton---#
        img_skeleton=branch_sklenton(img)

        # !!!!!
        # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 4))
        # ax[0,0].imshow(img, cmap=cm.gray)
        # ax[0,0].set_title('ori')  # only adaptive threshold
        # ax[0,1].imshow(img_skeleton, cmap=cm.gray)
        # ax[0,1].set_title('fiber')  # auto threshold
        # ax[0,2].imshow(img_ridge, cmap=cm.gray)
        # ax[0,2].set_title('fiber+synapse')  # detector ridge like region
        # ax[1,0].imshow(img_label)
        # ax[1,0].set_title('labeled(manual)')
        # plt.show()
        # !!!!!

        #img = hessian(img, **kwargs)
        features = features_func(img)
        result = future.predict_segmenter(features, clf)

        img_ridge_t = img_ridge.copy()
        #cimg=hessian(img,**kwargs)
        print('result label', set(list(result.ravel())))
        img_ridge_t[result != 3] = 0
        img_binary=synapse_roi_mask(img_ridge_t,result)
        labeled_img, _ = ndi.label(img_binary)
        image_label_overlay = label2rgb(labeled_img, image=img, bg_label=0,image_alpha=1)
        synapse_contour(img_binary)
        contours = measure.find_contours(img_binary, 0.8)

        # Display the image and plot all contours found
        fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(img_binary, cmap=plt.cm.gray)

        #contours=contour_filter(contours)
        contours=synapse_contour(img_binary)
        for n, contour in enumerate(contours):
            
            # hull = ConvexHull(contour)
            # # print('***',contour)
            # # print(hull.points)
            # points=np.array(hull.points)
            # ax[0].plot(points[:,1],points[:,0],linewidth=2)
            ax[0].plot(contour[:,:, 0], contour[:,:,1], linewidth=2)
        print(n)

        ax[1].imshow(img,cmap='gray')
        ax[2].imshow(img_label)
        plt.show()



        fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 4))
        for cnt in contours:
            colo=random.sample(DEFAULT_COLORS, 1)[0]
            #print(colo)
            image_label_overlay=cv2.drawContours(image_label_overlay, [cnt], -1, colo, -1)
        #ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
        #ax[0].contour(training_labels)
        ax[0,0].imshow(img,cmap='gray')
        ax[0,0].set_title('ori')
        ax[0,1].set_title('ori mask')
        ax[0,1].imshow(img_label,cmap='gray')
        ax[0,2].imshow(result,cmap='gray')
        ax[0,2].set_title('Segmentation')
        ax[1,0].imshow(img_ridge_t,cmap='gray')
        ax[1,0].set_title('result mask')
        ax[1,1].imshow(img_binary,cmap='gray')
        ax[1,1].set_title('final mask')
        ax[1,2].imshow(image_label_overlay)
        ax[1,2].set_title('final label')
        fig.tight_layout()

        plt.show()
        # if expand_pixel:
        #     new_contours=[]
        #     for contour in contours:
        #         contour=func_base.expand_polygon(contour,expand_pixel)
        #         new_contours.append(contour)
        # else:
        #     new_contours=contours
        write_roi(contours,outname,img)
    
    # shift_roi.reverse_shift(targetlist,outlist)

def main():
    outfolder='F:/LHJ/new'
    template='F:/LHJ/cell2/analysis/MAX_Stablized MAX_Stabilized_cell2-base-0mg_CSU-561(2).tif'
    target_list=['F:/LHJ/cell1/MAX_Stablized MAX_Stabilized_cell3-base-0mg-GPT_CSU-561(2).tif']
    outlist=["F:/LHJ/new/MAX_Stablized MAX_Stabilized_cell3-base-0mg-GPT_CSU-561(2).tif"]

if __name__ == "__main__":
    main()