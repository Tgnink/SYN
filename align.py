#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   align.py
@Time    :   2020/12/22 19:58:33
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, LNP2 group
@Desc    :   overlap two image with maximum overlap origon based on moephology
'''

# here put the import lib


import random
from matplotlib import patches
from skimage.feature import corner, match_template
from scipy import stats
from tifffile import imread, imwrite,TiffWriter
from re import template
from skimage.exposure import match_histograms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import roifile
from roifile import ImagejRoi
import os
import zipfile

from tifffile.tifffile import main
def show_hist_match_result(imagesource,imagetemplate,matched):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(imagesource)
    ax1.set_title('Source')
    ax2.imshow(imagetemplate)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')  


    plt.tight_layout()
    plt.show()

def caculate_shift(imagetemplate,matched,sublength=60,maxshift=10,minroilength=100):
    subwideth=sublength
    subheight=sublength
    maxshiftx=maxshift
    maxshifty=maxshift
    minroix=minroilength
    minroiy=minroilength
    height,width=imagetemplate.shape
    size=11#height//minroiy
    seedx=np.random.randint(subwideth,width-subwideth-minroix,size=size)
    seedy=np.random.randint(subheight,height-subheight-minroiy,size=size)
    shiftx=[]
    shifty=[]
    rectangles=[]

    for starx,stary in zip(seedx,seedy):
        roi=[starx,stary,starx+minroix,stary+minroiy]
        roiimage=imagetemplate[roi[1]:roi[3],roi[0]:roi[2]]
        #print('shape',roiimage.shape,roi)
        fig1=imagetemplate[roi[1]-maxshifty:roi[3]+maxshifty,roi[0]-maxshiftx:roi[2]+maxshifty]
        fig2=matched[roi[1]-maxshifty:roi[3]+maxshifty,roi[0]-maxshiftx:roi[2]+maxshifty]

        result1 = match_template(fig1,roiimage)
        ij = np.unravel_index(np.argmax(result1), result1.shape)
        result2 = match_template(fig2,roiimage)
        ji = np.unravel_index(np.argmax(result2), result2.shape)
        # if abs(ji[0])+abs(ji[1])>20:
        #     ax1.imshow(roiimage)
        #     ax2.imshow(imagetemplate)
        #     ax3.imshow(matched)
        #     plt.show()
        rectangles.append([starx,stary,starx-ij[1]+ji[1],stary-ij[0]+ji[0]])

        #print(ij,ji,ij[0]-ji[0],ij[1]-ji[1])
        shiftx.append(ij[1]-ji[1])
        shifty.append(ij[0]-ji[0])
    
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),sharex=True,sharey=True)
    # ax1.imshow(imagetemplate)
    # ax2.imshow(matched)
    # for roirec in rectangles:
    #     #print('www')
    #     x1,y1,x2,y2=roirec
    #     rect=plt.Rectangle((x1, y1), minroix, minroiy,edgecolor="red",facecolor='None', alpha=0.6)
    #     ax1.add_patch(rect)
    #     rect=plt.Rectangle((x2, y2), minroix, minroiy,edgecolor="blue",facecolor='None', alpha=0.6)
    #     ax1.add_patch(rect)
    #     rect=plt.Rectangle((x1, y1), minroix, minroiy,edgecolor="red",facecolor='None', alpha=0.6)
    #     ax2.add_patch(rect)
    #     rect=plt.Rectangle((x2, y2), minroix, minroiy,edgecolor="blue",facecolor='None', alpha=0.6)
    #     ax2.add_patch(rect)
    # plt.show()
    return shiftx,shifty

def analysis(template,targetlist,outlist,sublength=60,maxshift=20,minroilength=100):
    imagetemplate=Image.open(template)
    imagetemplate=np.asarray(imagetemplate)
    for imagesourcefile,outname in zip(targetlist,outlist):
        imagesource=Image.open(imagesourcefile)
        imagesource=np.asarray(imagesource)
        matched = match_histograms(imagesource, imagetemplate, multichannel=False)
        #show_hist_match_result(imagesource,imagetemplate,matched)
        np.random.seed(1)
        # correct shift image
        shiftx,shifty=caculate_shift(imagetemplate,matched,sublength,maxshift,minroilength)
        shift_x=stats.mode(shiftx)[0][0]
        shift_y=stats.mode(shifty)[0][0]
        print('x shift : ',shift_x,'\ny shift : ',shift_y)
        imagesource=np.roll(imagesource, shift_y, axis=0)
        imagesource=np.roll(imagesource, shift_x, axis=1)
        print('outfile : ',outname+'_align.tif')
        imwrite(outname+'_align.tif',imagesource)
        # correct shift roi
        roi2 = ImagejRoi.fromfile(template)
        outfilename=outname+"_align_roi.zip"
        if os.path.exists(outfilename):
            os.remove(outfilename)
        print('outfile:',outfilename)
        zip1 = zipfile.ZipFile(outfilename, 'w')
        zip1.close()
        #os.mkdir(os.path.join(pathway,name))
        for roi in roi2:
            roi.left=roi.left-shift_x
            roi.top=roi.top-shift_y
            roi.right=roi.right-shift_x
            roi.bottom=roi.bottom-shift_y
            roi.tofile(outfilename)









def test():
    # here load data
    imagefile1="F:\\LHJ\\new\\align_MIP_DFF_Stabilized_cell2-base-0mg-GPT-GPT+PEP_xf-CSU-488(0)_t0004.tif"
    imagefile2="F:\\LHJ\\new\\align_MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT_CSU-561(2).tif"

    imagesource=Image.open(imagefile1)
    imagetemplate=Image.open(imagefile2)

    imagesource=np.asarray(imagesource)
    imagetemplate=np.asarray(imagetemplate)

    # histgrom match
    matched = match_histograms(imagesource, imagetemplate, multichannel=False)
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
    #                                     sharex=True, sharey=True)
    # for aa in (ax1, ax2, ax3):
    #     aa.set_axis_off()

    # ax1.imshow(imagesource)
    # ax1.set_title('Source')
    # ax2.imshow(imagetemplate)
    # ax2.set_title('Reference')
    # ax3.imshow(matched)
    # ax3.set_title('Matched')  


    # plt.tight_layout()
    # plt.show()


    # save matched fig
    # from tifffile import imread, imwrite,TiffWriter
    # matched=np.uint16(matched)
    # matchfile="F:\\LHJ\\new\\align_new_MAX_Stabilized_cell2-base-0mg_xf-CSU-488(0)_t0000.tif"
    # # with TiffWriter('temp.tif') as tif:
    # #     tif.write(imagesource,photometric='minisblack')
    # #     tif.write(matched,photometric='minisblack')
    # print(matched.dtype,matched.shape,np.min(matched))
    # imwrite(matchfile,matched)

    # smooth fig

    diffimg=matched-imagesource


    # seed roi random and image template

    np.random.seed(1)  
    subwideth=60
    subheight=60
    height,width=imagetemplate.shape
    maxshiftx=20
    maxshifty=20
    minroix=100
    minroiy=100
    size=10#height//minroiy
    seedx=np.random.randint(subwideth,width-subwideth-minroix,size=size)
    seedy=np.random.randint(subheight,height-subheight-minroiy,size=size)

    shiftx=[]
    shifty=[]
    rectangles=[]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),sharex=True,sharey=True)
    for starx,stary in zip(seedx,seedy):
        roi=[starx,stary,starx+minroix,stary+minroiy]
        roiimage=imagetemplate[roi[1]:roi[3],roi[0]:roi[2]]
        print('shape',roiimage.shape,roi)
        fig1=imagetemplate[roi[1]-maxshifty:roi[3]+maxshifty,roi[0]-maxshiftx:roi[2]+maxshifty]
        fig2=matched[roi[1]-maxshifty:roi[3]+maxshifty,roi[0]-maxshiftx:roi[2]+maxshifty]

        result1 = match_template(fig1,roiimage)
        ij = np.unravel_index(np.argmax(result1), result1.shape)
        result2 = match_template(fig2,roiimage)
        ji = np.unravel_index(np.argmax(result2), result2.shape)
        # if abs(ji[0])+abs(ji[1])>20:
        #     ax1.imshow(roiimage)
        #     ax2.imshow(imagetemplate)
        #     ax3.imshow(matched)
        #     plt.show()
        rectangles.append([starx,stary,starx-ij[1]+ji[1],stary-ij[0]+ji[0]])

        print(ij,ji,ij[0]-ji[0],ij[1]-ji[1])
        shiftx.append(ij[1]-ji[1])
        shifty.append(ij[0]-ji[0])
    # ax1.imshow(imagetemplate)
    # ax2.imshow(matched)
    for roirec in rectangles:
        #print('www')
        x1,y1,x2,y2=roirec
        rect=plt.Rectangle((x1, y1), minroix, minroiy,edgecolor="red",facecolor='None', alpha=0.6)
        ax1.add_patch(rect)
        rect=plt.Rectangle((x2, y2), minroix, minroiy,edgecolor="blue",facecolor='None', alpha=0.6)
        ax1.add_patch(rect)
        rect=plt.Rectangle((x1, y1), minroix, minroiy,edgecolor="red",facecolor='None', alpha=0.6)
        ax2.add_patch(rect)
        rect=plt.Rectangle((x2, y2), minroix, minroiy,edgecolor="blue",facecolor='None', alpha=0.6)
        ax2.add_patch(rect)

    # plt.show()



    # correct shift

    shift_x=stats.mode(shiftx)[0][0]
    shift_y=stats.mode(shifty)[0][0]
    print('x shift : ',shift_x,'\ny shift : ',shift_y)
    imagesource=np.roll(imagesource, shift_y, axis=0)
    imagesource=np.roll(imagesource, shift_x, axis=1)
    # matchfile="F:\\LHJ\\new\\align_new_MAX_Stabilized_cell2-base-0mg_xf-CSU-488(0)_t0000.tif"
    # imwrite(matchfile,imagesource)

if __name__ == '__main__':
    test()
    

