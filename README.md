# SYN
drift roi correct for imagej

# SYN
drift roi correct for imagej
## requirenment for environgment
tifffile==2020.12.4
roifile==2020.11.28
matplotlib==3.2.2
opencv_contrib_python==4.4.0.46
scikit_image.egg==19.0 (the newest github of skimage)
numpy==1.18.5
exifread==2.3.2
Pillow==8.0.1
scikit_learn==0.23.2
skimage==0.0

you need install qt5 and opencv
# usage
it's a GUI script , you need prepare one image overlap with roi as template

provide 4 functions:
1. drift roi position , you will get a roi zip file
2. align two image , you will get a new image with max merge percent to template image. And if template has rois, will get new roi zip file
3. expand roi area
4. template to find new rois ( doesn't work)
