#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   branchs.py
@Time    :   2020/12/15 13:20:29
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''
from os import name
from numpy.lib.type_check import imag
from synapse_roi import main
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import diameter_closing, diameter_opening, opening, skeletonize, closing, area_closing, remove_small_objects, binary_opening,convex_hull_image
from PIL import Image, ImageDraw


class Branch(object):
    def __init__(self,length=None,label=None,chilnodes=None) -> None:
        self.childNodes=chilnodes
        self.length=length


def label_branch(label_map,skeleton_im):
    pass

def check_level(level_dict,label):
    for level,values in level_dict.items():
        if label in values:
            return level
def remove_label(level_dict,label):
    for level,values in level_dict.items():
        if label in values:
            level_dict[level].remove(label)
            return

def add_level(level_dict,label,level):
    if level in level_dict:
        level_dict[level].append(label)
    else:
        level_dict[level]=[label]
    


# here put the import lib
def analysis_branch(skeleton_im,root):
    # only for 2D , two value : false or true ; 0 or 1 ; 0 or 255
    label_map=np.zeros_like(skeleton_im,dtype=np.uint8)
    nodes_map=np.zeros_like(skeleton_im,dtype=np.uint8)
    flag=True
    height,weight=skeleton_im.shape
    if not skeleton_im[root[0],root[1]] :
        # if root is not white pixel need chose the neat white pixel as root
        radius=1 # search rectangle
        while flag:
            for y in range(-radius,radius+1):
                x=2*radius-abs(y)
                if skeleton_im[y,x]:
                    root=[y,x]
                    flag=False
                elif skeleton_im[y,-x]:
                    root=[y,x]
                    flag=False
    flag=True
    # mark and level branch
    # ini the root map
    # node ==-1
    levels={}
    name_labels={}
    name_labels[1]='1'
    levels[1]=[1]
    c_label=1
    max_label=1
    label_map[root[0],root[1]]=1
    current_poss=[np.array(root)]
    nodes=[] # store branch nodes
    count=0
    merges=[]
    text_pos={}
    text_pos[1]=root

    while current_poss:
        # check 8 pixel around current point
        next_poss=[]
        for current_pos in current_poss:
            next_pos=[]
            current_label=label_map[current_pos[0],current_pos[1]]
            py=[
                current_pos[0]-1,
                current_pos[0]-1,
                current_pos[0]-1,
                current_pos[0],
                current_pos[0],
                current_pos[0]+1,
                current_pos[0]+1,
                current_pos[0]+1]
            px=[
                current_pos[1]-1,
                current_pos[1],
                current_pos[1]+1,
                current_pos[1]-1,
                current_pos[1]+1,
                current_pos[1]-1,
                current_pos[1],
                current_pos[1]+1]
            for y,x in zip(py,px):
                value=skeleton_im[y,x]
                if value : # if pixel in branch skeleton
                    next_pos.append([y,x])
            if len(next_pos)==2: # no branch but not complet branch label
                merge=[]
                for pos in next_pos:
                    value=label_map[pos[0],pos[1]]
                    if not value:
                        label_map[pos[0],pos[1]]=current_label
                        next_poss.append(pos)
                    else:
                        merge.append(value)
                if len(merge)>1 and merge not in merges:
                    merges.append(merge)
            
            elif len(next_pos)<2: # complet branch label
                for pos in next_pos:
                    value=label_map[pos[0],pos[1]]
                    if value:
                        continue
                    label_map[pos[0],pos[1]]=current_label
                    next_poss.append(pos)
            elif len(next_pos)>2: # branch node
                count=count+1
                node=[current_pos[0],current_pos[1]]
                label_flag=False
                if  node not in nodes:
                    #label_map[node[0],node[1]]=0
                    #skeleton_im[node[0],node[1]]=0
                    nodes.append(node) # add branch node
                    nodes_map[node[0],node[1]]=np.max(nodes_map[node[0]-1:node[0]+2,node[1]-1:node[1]+2])+1
                # update level dict
                current_level=check_level(levels,current_label)
                
                for pos in next_pos:
                    value=label_map[pos[0],pos[1]]
                    if value == current_label:
                        label_flag=True
                if not label_flag:
                    remove_label(levels,current_label)
                    current_level=current_level-1
                    
                for pos in next_pos:
                    value=label_map[pos[0],pos[1]]
                    if value:
                        continue
                    max_label=max_label+1
                    label_map[pos[0],pos[1]]=max_label
                    text_pos[max_label]=pos
                    add_level(levels,max_label,current_level+1)
                    next_poss.append(pos)
               

        if len(next_poss)==0:
            print(0)
            pass
        current_poss=next_poss

    # remove node label
    for node in nodes:
        label_map[node[0],node[1]]=0
    # resort node
    
    # relabel map ,remove one branch has multilabel
    for merge in merges:
        min_label=min(merge)
        max_label=max(merge)
        label_map[label_map==max_label]=min_label
        level_min=check_level(levels,min_label)
        level_max=check_level(levels,max_label)
        del_label=min_label if level_min>level_max else max_label
        remove_label(levels,del_label)
    print(levels)
    #label_map=np.max(label_map,skeleton_im*255)
    

    print('branch number ',len(set(label_map.ravel())))
    print('label',set(label_map.ravel()))
    print('merge',merges)
    print(nodes)

    plt.figure()
    plt.imshow(label_map,cmap='jet')
    
    plt.colorbar()
    for labels in levels.values():
        for label in labels:
            print(label,len(np.argwhere(label_map==label)))
            pos=text_pos[label]
            posy=np.mean(np.where(label_map==label)[0])
            posx=np.mean(np.where(label_map==label)[1])
            plt.text(posx, posy, str(label), fontsize=20,color = "r")
    return label_map,nodes_map
                

def test():
    print('jjj')
    imagefile='ske.png'
    img=Image.open(imagefile)
    print(img.format, img.size, img.mode)
    img = img.convert('L')
    img = np.asarray(img)
    img=skeletonize(img>0)
    print(img)


    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 4))

    label_map,node_map=analysis_branch(img,[40,107])
    ax[0].imshow(img,cmap='gray')
    label_map[label_map!=0]=label_map[label_map!=0]*2+10
    ax[1].imshow(label_map,cmap='jet')
    aa=ax[2].imshow(node_map,cmap='gray')
    plt.colorbar(aa)

    
    
    plt.show()




if __name__ == "__main__":
    test()












