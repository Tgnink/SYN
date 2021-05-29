#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   readroi.py
@Time    :   2020/11/22 01:06:04
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, LNP2 group
@Desc    :   None
'''

# here put the import lib
import os
import zipfile
<<<<<<< HEAD
from PyQt5.QtCore import Qt
import roifile
from roifile import ImagejRoi
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from skimage.draw import polygon
import numpy as np
=======

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
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from skimage import data
from skimage.feature import corner, match_template
from skimage.exposure import match_histograms
import align
from scipy import stats
<<<<<<< HEAD
from tifffile import TiffWriter
from PyQt5.QtWidgets import QProgressDialog,QMessageBox
=======
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
METHODS=['cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF_NORMED']
meth=methods[5]



def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        if fig.scroll_status == False:
            return
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        # print(event.button)
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            # print (event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        plt.draw()  # force re-draw

    def press(event):
        print('press', event.key)
        if event.key == 'control':
            fig.scroll_status = True

    def release(event):
        print('press', event.key)
        if event.key == 'control':
            fig.scroll_status = False

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.scroll_status = False
    fig.canvas.mpl_connect('key_press_event', press)
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    fig.canvas.mpl_connect('key_release_event', release)

    # return the function
    return zoom_fun


def draw_roi(rois, im):
    # draw = ImageDraw.Draw(im)
    for i in range(len(rois)):
        coord = rois[i].coordinates()
        ax.plot(coord[:, 0], coord[:, 1])
        # polygon(im, roi2[i].coordinates(),shape=((255,255,255),(255,255,255)))
    # del draw
    return im
<<<<<<< HEAD
def draw_roi_onimage(rois, im):
    # draw = ImageDraw.Draw(im)
    for i in range(len(rois)):
        coord = rois[i].coordinates()
        im=cv2.polylines(im, [coord], True, 65530, 1)
    return im
=======

>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
def _get_expand_XY(expand_pixel,expand_f,w,h):
    if expand_pixel:
        expand_pixel=int(expand_pixel)
        expand_x=expand_pixel
        expand_y=expand_pixel
    elif expand_f:
        expand_x=round(w *expand_f)
        expand_y=round(h * expand_f)
    return expand_x,expand_y
def get_roi_image(roi, im,expand_pixel=5,expand_f=0.5):
    # roi shape = (?,2) that's x,y
    # roi to rectangle
    height,weight=im.shape
    if isinstance(roi, roifile.roifile.ImagejRoi):
        x1,y1,x2,y2=roi.left, roi.top, roi.right, roi.bottom
        coord = roi.coordinates()
    elif isinstance(roi, np.ndarray):
        coord = roi
        x1, y1 = np.min(coord, axis=0)
        x2, y2 = np.max(coord, axis=0)
    # return im_roi, 4corner , w,h
    w=x2-x1
    h=y2-y1
    if expand_pixel:
        expand_pixel=int(expand_pixel)
        expand_x=expand_pixel
        expand_y=expand_pixel
    elif expand_f:
        expand_x=round(w *expand_f)
        expand_y=round(h * expand_f)
<<<<<<< HEAD
    # centerx=(x1+x2)/2
    # centery=(y1+y2)/2

    height,weight=im.shape  

    # out of range , need shrink tarimage
    out_left=0#min(x1-expand_x,0)
    out_right=0#min(weight-x2-expand_x,0)
    out_top=0#min(y1-expand_y,0)
    out_bottom =0#min(height-expand_y-y2,0)
=======
    centerx=(x1+x2)/2
    centery=(y1+y2)/2

    height,weight=im.shape   
    # out of range , need shrink tarimage
    out_left=min(x1-expand_x,0)
    out_right=min(weight-x2-expand_x,0)
    out_top=min(y1-expand_y,0)
    out_bottom =min(height-expand_y-y2,0)
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    # padding zero for out of edge 
    expand_left=expand_x+out_left
    expand_right=expand_x+out_right
    expand_top=expand_y+out_top
    expand_bottom=expand_y+out_bottom
<<<<<<< HEAD
    img_roi=im[y1-expand_top:y2+expand_bottom, x1-expand_left:x2+expand_right]

    # img_targ=np.zeros((y2-y1+2*expand_y,x2-x1+2*expand_x))
    # hh,ww=img_targ.shape
    # #print('kkkkkk',img_targ.shape,-out_top,hh+out_bottom,-out_left,ww+out_right)
    # img_targ[-out_top:hh+out_bottom,-out_left:ww+out_right]=im[y1-expand_top:y2+expand_bottom, x1-expand_left:x2+expand_right]
    # return roi image(maybe add flank expand edge) and 4 corner of expand roi image

    
    return img_roi, [x1-expand_left,x2+expand_right,y1-expand_top,y2+expand_bottom,x2-x1+expand_right+expand_left,y2+expand_bottom+expand_top-y1]
=======
    img_targ=np.zeros((y2-y1+2*expand_y,x2-x1+2*expand_x))
    hh,ww=img_targ.shape
    #print('kkkkkk',img_targ.shape,-out_top,hh+out_bottom,-out_left,ww+out_right)
    img_targ[-out_top:hh+out_bottom,-out_left:ww+out_right]=im[y1-expand_top:y2+expand_bottom, x1-expand_left:x2+expand_right]
    # return roi image(maybe add flank expand edge) and 4 corner of expand roi image

    
    return img_targ, [x1-expand_x,x2+expand_x, y1-expand_y,y2+expand_y,x2-x1+2*expand_x,y2-y1+2*expand_y],[0,0]
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01

    # if edge out of canvas
    outedgex=min(x1-expand_x,weight-x2-expand_x)
    outedgey=min(y1-expand_y,height-expand_y-y2)
    if outedgex<0:
        expand_x=expand_x+outedgex
    if outedgey<0:
        expand_y=outedgey+expand_y
    if outedgex<0 or outedgey<0:
        print('coord:',x1-expand_x,x2+expand_x, y1-expand_y,y2+expand_y,x2-x1+2*expand_x,y2-y1+2*expand_y)


    # return roi image(maybe add flank expand edge) and 4 corner of expand roi image , W,H
    return im[y1-expand_y:y2+expand_y, x1-expand_x:x2+expand_x], [x1-expand_x,x2+expand_x, y1-expand_y,y2+expand_y,x2-x1+2*expand_x,y2-y1+2*expand_y],[0,0]
def get_target_image(im,cornor,expand_pixel=5,expand_f=0):
    x1,x2,y1,y2,w,h=cornor
<<<<<<< HEAD
    #expand_x,expand_y=_get_expand_XY(expand_pixel,expand_f,w,h)
    height,weight=im.shape   
    # out of range , need shrink tarimage
    # out_left=min(x1-expand_x,0)
    # out_right=min(weight-x2-expand_x,0)
    # out_top=min(y1-expand_y,0)
    # out_bottom =min(height-expand_y-y2,0)
    # padding zero for out of edge 
    expand_left=expand_pixel
    expand_right=expand_pixel
    expand_top=expand_pixel
    expand_bottom=expand_pixel
    img_targ=im[y1-expand_top:y2+expand_bottom, x1-expand_left:x2+expand_right]
    
    # return roi image(maybe add flank expand edge) and 4 corner of expand roi image
    return img_targ, [x1-expand_left,x2+expand_right,y1-expand_top,y2+expand_bottom,x2-x1+expand_right+expand_left,y2+expand_bottom+expand_top-y1]
=======
    expand_x,expand_y=_get_expand_XY(expand_pixel,expand_f,w,h)
    height,weight=im.shape   
    # out of range , need shrink tarimage
    out_left=min(x1-expand_x,0)
    out_right=min(weight-x2-expand_x,0)
    out_top=min(y1-expand_y,0)
    out_bottom =min(height-expand_y-y2,0)
    # padding zero for out of edge 
    expand_left=expand_x+out_left
    expand_right=expand_x+out_right
    expand_top=expand_y+out_top
    expand_bottom=expand_y+out_bottom
    img_targ=np.zeros((y2-y1+2*expand_y,x2-x1+2*expand_x))
    hh,ww=img_targ.shape
    #print('kkkkkk',img_targ.shape)
    img_targ[-out_top:hh+out_bottom,-out_left:ww+out_right]=im[y1-expand_top:y2+expand_bottom, x1-expand_left:x2+expand_right]
    # return roi image(maybe add flank expand edge) and 4 corner of expand roi image
    return img_targ, [x1-expand_x,x2+expand_x, y1-expand_y,y2+expand_y,x2-x1+2*expand_x,y2-y1+2*expand_y]
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
def positive(llist):
    for n,v in enumerate(llist):
        if v<0:
            llist[n]=0
    return llist

<<<<<<< HEAD
def match_roi(roi_im, target_im, corner_roi, corner_tar,align_x=0,align_y=0,meth=None):
=======
def match_roi(roi_im: np.ndarray, target_im: np.ndarray, x0: int, y0: int,align_x=0,align_y=0,meth=None) -> list:
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    '''
    :param roi_im:
    :param im: im must large than roi_im
    :param x0: offset
    :param y0: offset
    :return:
    '''
    # print(target_im.shape,roi_im.shape)
    if not meth:
        meth=METHODS[0]
    method = eval(meth)
<<<<<<< HEAD
    # print(target_im.shape,roi_im.shape)
    res = cv2.matchTemplate(np.array(target_im,dtype=np.float32), np.array(roi_im,dtype=np.float32), method)

=======
    res = cv2.matchTemplate(np.array(target_im,dtype=np.float32), np.array(roi_im,dtype=np.float32), method)
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # result = match_template(target_im, roi_im)
    # ij = np.unravel_index(np.argmax(result), result.shape)
    # y,x = ij
<<<<<<< HEAD
    if meth=='cv2.TM_SQDIFF_NORMED':
        x,y=min_loc
    else:
        x,y=max_loc#top_left = max_loc 
    # print('???',(y-ij[0],x-ij[1]),ij,res.shape,result.shape)
    h,w=roi_im.shape
    x0=corner_tar[0]
    y0=corner_tar[2]
=======
    x,y=min_loc
    # print('???',(y-ij[0],x-ij[1]),ij,res.shape,result.shape)
    h,w=roi_im.shape
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    height,weight=target_im.shape
    x1=x0+x # if x == expand_pixel that's ori pos no shift
    x2=x0+x+w
    y1=y0+y
    y2=y0+y+h
    # campare align result , if align roi has higner intensity refuse it ,receive new
    # if align_x or align_y:
    #     shiftintensity=np.sum(target_im[y1-y0:y2-y0,x1-x0:x2-x0])
    #     alignintensity=np.sum(target_im[align_y:align_y+h,align_x:align_x+w])
    #     if alignintensity>shiftintensity:
    #         return target_im[align_y:align_y+h,align_x:align_x+w],[x0+align_x,x0+w+align_x,y0+align_y,y0+align_y+h,w,h]
    # return center and new roi
<<<<<<< HEAD
    # c=y1-y0
    # d=height-y2+y0
    # if x<0 or y<0 or c<0 or d<0:
    #     print(x,y,c,d,"error: out of target range")
    #     return False,[x1,x2,y1,y2,x2-x1,y2-y1]
    return target_im[y:y+h,x:x+w],[x1,x2,y1,y2,x2-x1,y2-y1]
=======
    c=y1-y0
    d=height-y2+y0
    if x<0 or y<0 or c<0 or d<0:
        print(x,y,c,d,"error: out of target range")
        return False,[x1,x2,y1,y2,x2-x1,y2-y1]
    return target_im[y1-y0:y2-y0,x1-x0:x2-x0],[x1,x2,y1,y2,x2-x1,y2-y1]
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01

def get_re_roi_image(temp1,temp2,img,method=2):

    if method==1:
        x1,x2,y1,y2,w,h=temp2
        height,width=img.shape
        if x1 < 0:
            x2 = x2 - x1
            x1 = 0
        if x2 > width:
            x1 = x1 - x2 + width - 1
            x2 = width - 1
        if y1 < 0:
            y2 = y2 - y1
            y1 = 0
        if y2 > height:
            y1 = y1 - y2 + height - 1
            y2 = height - 1

        im_roi = img[y1:y2, x1:x2]
        return  im_roi,[x1,y2,y1,y2,w,h]
    elif method==2:
        x1, x2, y1, y2,w,h = temp1
        im_roi= img[y1:y2, x1:x2]
        return  im_roi,[x1,x2,y1,y2,w,h]

<<<<<<< HEAD
def _get_template_img_from_roi(image_template,roi2,expand_pixel=5,expand_f=0):
    roi_corners = {}
    roi_imgs={}
    roi_offsets={}
    # img = Image.open(imagefile)
    # print(img.format, img.size, img.mode)
    # img = np.asarray(img)
    height,width=image_template.shape
=======
def _get_roiimg_from_roi(imagefile,roi2,expand_pixel=5,expand_f=0):
    roi_corners = {}
    roi_imgs={}
    roi_offsets={}
    img = Image.open(imagefile)
    print(img.format, img.size, img.mode)
    img = np.asarray(img)
    height,width=img.shape
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    for roi in roi2:
        if roi.name in roi_corners:
            print('error',roi.name)
            roi.name=roi.name+'404'
        roi_corners[roi.name]=[]
        roi_imgs[roi.name]=None
        roi_offsets[roi.name]=[]
<<<<<<< HEAD
        im_roi, [x1, x2, y1, y2,w,h]=get_roi_image(roi,image_template,expand_pixel)
        roi_corners[roi.name].append([x1, x2, y1, y2,w,h])
        roi_imgs[roi.name]=im_roi
        roi_offsets[roi.name].append([0,0])
=======
        im_roi, [x1, x2, y1, y2,w,h],[shiftx,shifty]=get_roi_image(roi,img,expand_pixel)
        roi_corners[roi.name].append([x1, x2, y1, y2,w,h])
        roi_imgs[roi.name]=im_roi
        roi_offsets[roi.name].append([shiftx,shifty])
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01

        # -- reget roi
        # x1, x2, y1, y2, w, h = roi_corners[roi.name][0]
        # temp1=[x1, x2, y1, y2, w, h]
        # if expand_pixel:
        #     expand_x=expand_pixel
        #     expand_y=expand_pixel
        # elif expand_f:
        #     expand_x=round(w *expand_f)
        #     expand_y=round(h * expand_f)
        # x01, x02, y01, y02 = positive([x1 - expand_x, x2 + expand_x, y1 - expand_y , y2 + expand_y])
        # target_im = img[y01:y02, x01:x02]
        # im_roi = roi_imgs[roi.name]
        # im_roi, [x1, x2, y1, y2, w, h] = match_roi(im_roi, target_im, x01, y01)

        # if im_roi is False:

        #     temp2=[x1, x2, y1, y2, w, h]
        #     im_roi,[x1, x2, y1, y2, w, h]=get_re_roi_image(temp1,temp2,img,method=1)

        #     roi_corners[roi.name][0]=([x1, x2, y1, y2, w, h])
        #     roi_imgs[roi.name] = im_roi
    return roi_corners,roi_imgs,roi_offsets
<<<<<<< HEAD
def _get_roiimg_from_match(img,image_template,roi_corners,roi_imgs,roi_offsets,roi2,expand_pixel=5,expand_f=0,meth=None):
    img = np.asarray(img)
    height,width=img.shape
    outofborder=0


    for roi in roi2:
        corner1 = roi_corners[roi.name][-1]
        #corner1=[x1, x2, y1, y2, w, h]
        # expand_x,expand_y=_get_expand_XY(expand_pixel,expand_f,corner1[-2],corner1[-1])
        target_im,cornor2=get_target_image(img,corner1,expand_pixel,expand_f)
        template_roi = roi_imgs[roi.name]

        try:
            new_template_roi, corner3= match_roi(template_roi,target_im,corner1,cornor2,expand_pixel,expand_pixel,meth=meth)

        except :
            print(roi_corners[roi.name],cornor2)
            print(template_roi.shape,target_im.shape)
            roi_imgs[roi.name]=template_roi
            roi_corners[roi.name].append(cornor2)
            shiftx=0
            shifty=0
            roi_offsets[roi.name].append([shiftx,shifty])
            continue


        # if template_roi is False:
        #     outofborder+=1
        #     template_roi,corner3=get_re_roi_image(corner1,corner3,img,method=1)
        #     print(roi.__hash__(),corner1,cornor2,corner3)
        roi_imgs[roi.name]=new_template_roi
        roi_corners[roi.name].append(corner3)
        shiftx=round(corner3[0]-corner1[0])
        shifty=round(corner3[2]-corner1[2])
        roi_offsets[roi.name].append([shiftx,shifty])


    #print('total out of border : ', outofborder)
=======
def _get_roiimg_from_match(imagefile,image_template,roi_corners,roi_imgs,roi_offsets,roi2,expand_pixel=5,expand_f=0,meth=None):
    img = Image.open(imagefile)
    print(img.format, img.size, img.mode)
    img = np.asarray(img)
    img = match_histograms(img, image_template, multichannel=False)
    height,width=img.shape
    outofborder=0

    alignx,aligny=align.caculate_shift(image_template,img)
    align_x=-stats.mode(alignx)[0][0]
    align_y=-stats.mode(aligny)[0][0]
    print('alignx',align_x,'align_y',align_y)
    for roi in roi2:
        corner1 = roi_corners[roi.name][-1]
        #corner1=[x1, x2, y1, y2, w, h]
        expand_x,expand_y=_get_expand_XY(expand_pixel,expand_f,corner1[-2],corner1[-1])
        target_im,cornor2=get_target_image(img,corner1,expand_pixel,expand_f)
        im_roi = roi_imgs[roi.name]

        try:
            im_roi, corner3= match_roi(im_roi,target_im,cornor2[0],cornor2[2],expand_pixel+align_x,expand_pixel+align_y,meth=meth)
        except :
            print(roi_corners[roi.name],cornor2)
            print(im_roi.shape,target_im.shape)
            raise

        if im_roi is False:
            outofborder+=1
            im_roi,corner3=get_re_roi_image(corner1,corner3,img,method=1)
            print(roi.__hash__(),corner1,cornor2,corner3)
        roi_imgs[roi.name]=im_roi
        roi_corners[roi.name].append(corner3)
        shiftx=round(((corner3[0]+corner3[1])-(corner1[0]+corner1[1]))/2)
        shifty=round(((corner3[2]+corner3[3])-(corner1[2]+corner1[3]))/2)
        roi_offsets[roi.name].append([shiftx,shifty])


    print('total out of border : ', outofborder)
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
    return roi_corners,roi_imgs,roi_offsets


def reverse_shift(imagelist,roifilelist,expand_f=2):
    # calculate max roi to first image
    outname=imagelist[0]
    path,outname=os.path.split(outname)
    outname, postfix = os.path.splitext(outname)
    path,roiname=os.path.split(roifilelist[0])
    outname=os.path.join(path,outname)
    for n,(roiname,imagefile) in enumerate(zip(roifilelist,imagelist)):
        roi2 = ImagejRoi.fromfile(roiname+'_roi.zip')
<<<<<<< HEAD
        roi_corners,roi_imgs,roi_offsets=_get_template_img_from_roi(imagefile,roi2,expand_f)
=======
        roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_roi(imagefile,roi2,expand_f)
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
        for i in range(n-1,-1,-1):
            roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_match(imagelist[i],roi_corners,roi_imgs,roi2,roi_offsets,expand_f)
        filename=outname+'_part'+str(n)+'.zip'
        print('name', filename)
        print('roi',roiname+'_roi.zip')
        if os.path.exists(filename):
            os.remove(filename)
        zip1 = zipfile.ZipFile(filename, 'w')
        zip1.close()
        #os.mkdir(os.path.join(pathway,name))
        for roi in roi2:
            x1, x2, y1, y2, w, h =roi_corners[roi.name][-1]
            roi.left=x1
            roi.top=y1
            roi.right=x2
            roi.bottom=y2
            roi.tofile(filename)
<<<<<<< HEAD
        #replicate=cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REPLICATE)
def analysis(imagelist,roiname,outlist,expand_pixel,roi_enlarge_pixel,expand_f=0,meth=None):
    roi2 = ImagejRoi.fromfile(roiname)
    image_template=Image.open(roiname)
    image_template=np.asarray(image_template,dtype=np.float32)

    for imagefile,outname in zip(imagelist,outlist):
        print('processing file:',imagefile)
        img = Image.open(imagefile)
        print(img.format, img.size, img.mode)
        img = np.asarray(img,dtype=np.float32)
        img = match_histograms(img, image_template, multichannel=False)
        height,width=img.shape
        outofborder=0

        # align first
        alignx,aligny=align.caculate_shift(image_template,img)

        alignx=stats.mode(alignx)[0][0]
        aligny=stats.mode(aligny)[0][0]
        print('alignx',-alignx,'align_y',-aligny)
        img=np.roll(img, aligny, axis=0)
        img=np.roll(img, alignx, axis=1)

        # padding edge ,top,bottom,left,right
        # expand pixel
        print(image_template.shape)
        image_template=cv2.copyMakeBorder(image_template,expand_pixel,expand_pixel,expand_pixel,expand_pixel,borderType=cv2.BORDER_REPLICATE)
        img=cv2.copyMakeBorder(img,expand_pixel,expand_pixel,expand_pixel,expand_pixel,borderType=cv2.BORDER_REPLICATE)
        print(image_template.shape)
        for roi in roi2:
            a,b,c,d=roi.left,roi.bottom,roi.right,roi.top 
            if a<0 or b<0 or c>=width or d>=height:
                print('err',a,b,c,d)

            roi.left=roi.left+expand_pixel
            roi.top=roi.top+expand_pixel
            roi.right=roi.right+expand_pixel
            roi.bottom=roi.bottom+expand_pixel
            
        
        roi_corners,roi_imgs,roi_offsets=_get_template_img_from_roi(image_template,roi2,expand_pixel=roi_enlarge_pixel,expand_f=expand_f)

        # match template
        pre_image=image_template
        roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_match(img,pre_image,roi_corners,roi_imgs,roi_offsets,roi2,expand_pixel,expand_f,meth)
        pre_image=Image.open(imagefile)
        pre_image=np.asarray(pre_image)

        # write new roi
        outfilename=outname+"_shift.zip"
=======
        
def analysis(imagelist,roiname,outlist,expand_pixel,roi_enlarge_pixel,expand_f=0,meth=None):
    roi2 = ImagejRoi.fromfile(roiname)
    image_template=Image.open(roiname)
    image_template=np.asarray(image_template)
    # expand pixel
    roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_roi(roiname,roi2,expand_pixel=roi_enlarge_pixel,expand_f=expand_f)
    pre_image=image_template
    for imagefile in imagelist:
        print('processing file:',imagefile)
        roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_match(imagefile,pre_image,roi_corners,roi_imgs,roi_offsets,roi2,expand_pixel,expand_f,meth)
        pre_image=Image.open(imagefile)
        pre_image=np.asarray(pre_image)
    for n,filename in enumerate(outlist):
        outfilename=filename+"_shift.zip"
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
        if os.path.exists(outfilename):
            os.remove(outfilename)
        print('outfile:',outfilename)
        zip1 = zipfile.ZipFile(outfilename, 'w')
        zip1.close()
        #os.mkdir(os.path.join(pathway,name))
        for roi in roi2:
<<<<<<< HEAD
            shiftx,shifty =roi_offsets[roi.name][1] # 0 is roi shift
            shiftx=int(shiftx)
            shifty=int(shifty)
            roi.left=roi.left+shiftx-expand_pixel-alignx
            roi.top=roi.top+shifty-expand_pixel-aligny
            roi.right=roi.right+shiftx-expand_pixel-alignx
            roi.bottom=roi.bottom+shifty-expand_pixel-aligny
            roi.tofile(outfilename)

def analysis_txt(widget,imagelist,roiname,outlist,expand_pixel,roi_enlarge_pixel,expand_f=0,meth=None):


    num = int(len(outlist))
    """
    我们新建了一个QProgressDialog对象progress，设置它的标题、标签、增加取消的按钮。
    """
    progress = QProgressDialog(widget)
    progress.setWindowTitle("请稍等")  
    progress.setLabelText("正在操作...")
    progress.setCancelButtonText("取消")
    """
    如果任务的预期持续时间小于minimumDuration，则对话框根本不会出现。这样可以防止弹出对话框，快速完成任务。对于预期超过minimumDuration的任务，对话框将在minimumDuration时间之后或任何进度设置后立即弹出。如果设置为0，则只要设置任何进度，将始终显示对话框。 默认值为4000毫秒,即4秒。
    """
    progress.setMinimumDuration(5)#此属性保留对话框出现之前必须通过的时间。
    """
    此属性保留由模态小部件阻止的窗口。
    这个属性只对Windows有意义。 模态小部件防止其他窗口中的小部件获取输入。 该属性的值控制在窗口小部件可见时阻止哪些窗口。 窗口可见时更改此属性无效; 您必须首先hide（）小部件，然后再次show（）。
    默认情况下，此属性为Qt.NonModal。
    """
    progress.setWindowModality(Qt.WindowModal)
    """
    由上面我们知道：使用setMinimum() 和setMaximum() 或构造函数来设置操作中的“steps”数量，并在操作进行时调用setValue()。setRange(0,num)就是设置其最小和最大值，这里最小值0，最大值num，num是根据输入框中的数字确定的。
    """
    progress.setRange(0,num) 

    """
    setValue()该属性持有当前的进度。要使进度对话框按预期的方式工作，您应该首先将此属性设置为QProgressDialog的最大最小值， 您可以在中间调用setValue()任意次数。
    通过wasCanceled()判断我们是否按下取消按钮，如果按下则提示失败。若for循环顺利结束，执行else后的语句，表明成功
    """       


    roi2 = ImagejRoi.fromfile(roiname)
    outfilename=outlist[0]+"_shift.zip"
    if os.path.exists(outfilename):
        os.remove(outfilename)
    print('outfile:',outfilename)
    zip1 = zipfile.ZipFile(outfilename, 'w')
    zip1.close()
    #os.mkdir(os.path.join(pathway,name))
    for roi in roi2:
        roi.tofile(outfilename)
    img = Image.open(imagelist[0])
    img = np.asarray(img,dtype=np.float32)
    img_ori=img.copy()
    imgs=[]

    
    path,filename=os.path.split(outlist[0])
    mergefile=os.path.join(path,"_merge.tif")
    tifroifile=os.path.join(path,"batch_sequence.txt")
    img_ori=draw_roi_onimage(roi2,img_ori)
    imgs.append(np.array(img_ori,dtype=np.uint16))

    
    count=1
    for imagefile,outname,templatefile in zip(imagelist[1:],outlist[1:],imagelist[:-1]):
        progress.setValue(count) 
        if progress.wasCanceled():
            QMessageBox.warning(widget,"提示","操作失败") 
            break
        count+=1

        roi2 = ImagejRoi.fromfile(outfilename)
        image_template=Image.open(templatefile)
        image_template=np.asarray(image_template,dtype=np.float32)

        print('processing file:',imagefile)
        img = Image.open(imagefile)
        print(img.format, img.size, img.mode)
        img = np.asarray(img,dtype=np.float32)
        img_ori=img.copy()
        img = match_histograms(img, image_template, multichannel=False)
        height,width=img.shape
        outofborder=0

        # align first
        alignx,aligny=align.caculate_shift(image_template,img)

        alignx=stats.mode(alignx)[0][0]
        aligny=stats.mode(aligny)[0][0]
        print('alignx',-alignx,'align_y',-aligny)
        img=np.roll(img, aligny, axis=0)
        img=np.roll(img, alignx, axis=1)

        # padding edge ,top,bottom,left,right
        # expand pixel
        # print(image_template.shape)
        image_template=cv2.copyMakeBorder(image_template,expand_pixel,expand_pixel,expand_pixel,expand_pixel,borderType=cv2.BORDER_REPLICATE)
        img=cv2.copyMakeBorder(img,expand_pixel,expand_pixel,expand_pixel,expand_pixel,borderType=cv2.BORDER_REPLICATE)
        # print(image_template.shape)
        for roi in roi2:
            a,b,c,d=roi.left,roi.bottom,roi.right,roi.top 
            if a<0 or b<0 or c>=width or d>=height:
                print('err',a,b,c,d)

            roi.left=roi.left+expand_pixel
            roi.top=roi.top+expand_pixel
            roi.right=roi.right+expand_pixel
            roi.bottom=roi.bottom+expand_pixel
            
        
        roi_corners,roi_imgs,roi_offsets=_get_template_img_from_roi(image_template,roi2,expand_pixel=roi_enlarge_pixel,expand_f=expand_f)

        # match template
        pre_image=image_template
        roi_corners,roi_imgs,roi_offsets=_get_roiimg_from_match(img,pre_image,roi_corners,roi_imgs,roi_offsets,roi2,expand_pixel,expand_f,meth)
        pre_image=Image.open(imagefile)
        pre_image=np.asarray(pre_image)

        # write new roi
        outfilename=outname+"_shift.zip"
        if os.path.exists(outfilename):
            os.remove(outfilename)
        print('outfile:',outfilename)
        zip1 = zipfile.ZipFile(outfilename, 'w')
        zip1.close()
        #os.mkdir(os.path.join(pathway,name))
        for roi in roi2:
            shiftx,shifty =roi_offsets[roi.name][1] # 0 is roi shift
            shiftx=int(shiftx)
            shifty=int(shifty)
            roi.left=roi.left+shiftx-expand_pixel-alignx
            roi.top=roi.top+shifty-expand_pixel-aligny
            roi.right=roi.right+shiftx-expand_pixel-alignx
            roi.bottom=roi.bottom+shifty-expand_pixel-aligny
            roi.tofile(outfilename)
        img_ori=draw_roi_onimage(roi2,img_ori)
        imgs.append(np.array(img_ori,dtype=np.uint16))
    else:
        progress.setValue(num)
        QMessageBox.information(widget,"提示","操作成功")
    # write roi and image merge
    tif= TiffWriter(mergefile) 
    for img in imgs:
        tif.write(img, contiguous=True)
    tif.close()
    # write batch txt for caculate roi
    with open(tifroifile,"w") as f:
        for imagefile,outname in zip(imagelist,outlist):
            outfilename=outname+"_shift.zip"
            f.write(imagefile+"\n")
            f.write(outfilename+"\n")
=======
            shiftx,shifty =roi_offsets[roi.name][n+1]
            shiftx=int(shiftx)
            shifty=int(shifty)
            roi.left=roi.left+shiftx
            roi.top=roi.top+shifty
            roi.right=roi.right+shiftx
            roi.bottom=roi.bottom+shifty
            roi.tofile(outfilename)



>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
def test():

    # -- 1. load fig
    plt.figure()
    ax = plt.subplot(111)
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)

    # -- 2. load roi and image

    pathway="F:\\LHJ\\cell2\\analysis"
    roiname='RoiSet-0mg.zip'
    roiname=os.path.join(pathway,roiname)
    roi2 = ImagejRoi.fromfile('F:\\LHJ\\cell2\\analysis\\MAX_Stablized MAX_Stabilized_cell2-base-0mg_CSU-561(2).tif')
    #roi2 = ImagejRoi.fromfile('F:\\LHJ\\cell2\\analysis\\RoiSet-0mg.zip')
    filelist=[
        "MAX_Stablized MAX_Stabilized_cell2-base-0mg_CSU-561(2).tif",
        "MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT_CSU-561(2).tif",
        "MAX_Stablized MAX_Stabilized_cell2-base-0mg-GPT-GPT+PEP_CSU-561(2).tif"
    ]
    #roi2[0].coordinates()
    roi_corners = {}
    roi_imgs={}
    expand_f=0.5
    for i,name in enumerate(filelist):
        filename=os.path.join(pathway,name)
        img = Image.open(filename)

        print(img.format, img.size, img.mode)
        img = np.asarray(img)
        height,width=img.shape
        # get roi_im from roi
        if i == 0:
            for roi in roi2:
                if roi.name in roi_corners:
                    print('error',roi.name)
                    roi.name=roi.name+'404'
                roi_corners[roi.name]=[]
                roi_imgs[roi.name]=None
                im_roi, [x1, x2, y1, y2,w,h]=get_roi_image(roi,img)
                # plt.figure()
                # ax = plt.subplot(111)
                # scale = 1.2
                # f = zoom_factory(ax, base_scale=scale)
                #
                # ax.imshow(im_roi)
                # roi.plot(ax)
                # plt.show()
                roi_corners[roi.name].append([x1, x2, y1, y2,w,h])
                roi_imgs[roi.name]=im_roi

                # -- reget roi
                x1, x2, y1, y2, w, h = roi_corners[roi.name][i]
                temp1=[x1, x2, y1, y2, w, h]
                x01, x02, y01, y02 = positive([x1 - round(w *expand_f), x2 + round(w *expand_f), y1 - round(h * expand_f), y2 + round(h * expand_f)])
                target_im = img[y01:y02, x01:x02]
                im_roi = roi_imgs[roi.name]
                im_roi, [x1, x2, y1, y2, w, h] = match_roi(im_roi, target_im, x01, y01)

                if im_roi is False:

                    temp2=[x1, x2, y1, y2, w, h]
                    im_roi,[x1, x2, y1, y2, w, h]=get_re_roi_image(temp1,temp2,img,method=1)

                    roi_corners[roi.name][0]=([x1, x2, y1, y2, w, h])
                    roi_imgs[roi.name] = im_roi

        else:
            outofborder=0
            for roi in roi2:

                x1, x2, y1, y2, w, h = roi_corners[roi.name][i-1]
                temp1=[x1, x2, y1, y2, w, h]
                x01,x02,y01,y02=positive([x1 - round(w *expand_f), x2 + round(w *expand_f), y1 - round(h * expand_f), y2 + round(h * expand_f)])
                target_im=img[y01:y02,x01:x02]
                im_roi = roi_imgs[roi.name]
                im_roi, [x1, x2, y1, y2,w,h]= match_roi(im_roi,target_im,x01,y01)
                if im_roi is False:
                    outofborder+=1
                    tem2=[x1, x2, y1, y2,w,h]
                    im_roi,[x1,x2,y1,y2,w,h]=get_re_roi_image(temp1,temp2,img,method=1)
                    print(roi.__hash__(),'   ',x1,x2,y1,y2,'   ',x01,x02,y01,y02)
                roi_imgs[roi.name]=im_roi
                roi_corners[roi.name].append([x1, x2, y1, y2, w, h])
            print('total out of border : ', outofborder)

    # -- 2. updata roi and save roi to zip
    import zipfile
    for i,name in enumerate(filelist):
        name,postfix=os.path.splitext(name)

        name=name+'_roi.zip'
        filename=os.path.join(pathway,name)
        print('name', name)
        if os.path.exists(filename):
            os.remove(filename)
        zip1 = zipfile.ZipFile(filename, 'w')
        zip1.close()
        #os.mkdir(os.path.join(pathway,name))
        for roi in roi2:
            x1, x2, y1, y2, w, h =roi_corners[roi.name][i]
            roi.left=x1
            roi.top=y1
            roi.right=x2
            roi.bottom=y2
            roi.tofile(filename)




if __name__ == "__main__":
    test()


# im shape is (h,w) that's (y,x)
# print(img.shape)
# ax.imshow(img)
# # im=im.convert("L")
# for i in range(2):
#     coord = roi2[i].coordinates()
#     ax.plot(coord[:, 0], coord[:, 1])

# im.show() cannot work , will call window picture browser
# image.show()函数会生成一个临时的.BMP文件用来显示，
# 所有使用什么软件显示是由.
# BMP图像默认打开软件决定的，
# 如果要更换image.show()显示软件，需要更换.BMP文件默认打开软件。

# plt.show()
