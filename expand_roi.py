#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   expand_roi.py
@Time    :   2020/12/12 20:28:26
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, LNP2 group
@Desc    :   None
'''
from numpy.lib.polynomial import poly
import roifile
import os
import zipfile
import numpy as np
# here put the import lib
def expand_polygon(polygon,pixel):
    # datatype : 0 roi 1 cnt 2 n*2 array
    datatype=0
    if isinstance(polygon,roifile.ImagejRoi):
        contour=polygon.integer_coordinates
        datatype=0
    elif isinstance(polygon,np.ndarray):
        # contour like [[[x,y]] in opencv
        # ontour like [[x,y]]
        lencontour=len(polygon.shape)
        if lencontour>2:
            contour=np.squeeze(polygon)
            datatype=1
        else:
            datatype=2
    else:
        contour=None
    top =np.min(contour[:,1])
    bottom=np.max(contour[:,1])
    left=np.min(contour[:,0])
    right=np.max(contour[:,0])
    #centerY=np.mean([top,bottom])
    #centerX=np.mean([right,left])
    centerY=np.mean(contour[:,1])
    centerX=np.mean(contour[:,0])
    new_contour=np.zeros_like(contour)
    for n,(x,y) in enumerate(contour):
        dy=y-centerY
        dx=x-centerX
        l=np.sqrt(dy*dy+dx*dx)
        deltay=(l+pixel)/l*dy
        deltax=(l+pixel)/l*dx
        newy=deltay+centerY
        newx=deltax+centerX
        new_contour[n,0]=int(newx)
        new_contour[n,1]=int(newy)
    if datatype==0:
        top =np.min(new_contour[:,1])
        bottom=np.max(new_contour[:,1])
        left=np.min(new_contour[:,0])
        right=np.max(new_contour[:,0])
        if top<0:
            polygon.top=polygon.top+top
            new_contour=new_contour+[0,-top]
        if left<0:
            polygon.left=polygon.left+left
            new_contour=new_contour+[-left,0]
        polygon.integer_coordinates=new_contour
        return polygon
    elif datatype==1:
        new_contour=new_contour[:,None,:]
        new_contour
    elif datatype==2:
        return new_contour

def analsis(roifilelist,outlist,expand_pixel):
    print('expand pixel',expand_pixel)
    for roiname,outname in zip(roifilelist,outlist):
        roi2 = roifile.ImagejRoi.fromfile(roiname)
        outname=outname+'_expand.zip'
        if os.path.exists(outname):
            os.remove(outname)
        print('out>',outname)
        zip1 = zipfile.ZipFile(outname, 'w')
        zip1.close()
        for n,roi in enumerate(roi2):
            roi=expand_polygon(roi,expand_pixel)
            roi.tofile(outname) 
