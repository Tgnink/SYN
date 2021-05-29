from skimage.draw import polygon
import numpy as np
img = np.zeros((10, 10), dtype=np.uint8)
r = np.array([1, 2, 8])
c = np.array([1, 7, 4])
rr, cc = polygon(r, c)
img[rr, cc] = 1+img[rr,cc]
r = np.array([3, 2, 8])
c = np.array([3, 7, 4])
rr, cc = polygon(r, c)
img[rr, cc] = 1+img[rr,cc]
print(img)