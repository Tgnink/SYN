from copy import Error
import os
import numpy as np
from numpy.core.fromnumeric import mean

def filelist(txtfile):
    try:
        f = open(txtfile, 'r')
        lines=f.readlines()
        file_list = []
        for line in lines:
            line=line.strip()
            file_list.append(line)
        f.close()
        return file_list
    except Error as e:
        print(e)

def expand_polygon(contour,pixel):
    # contour like [[[x,y]] in opencv
    # ontour like [[x,y]]
    lencontour=len(contour.shape)
    if lencontour>2:
        contour=np.squeeze(contour)
    top =np.min(contour[:,1])
    bottom=np.max(contour[:,1])
    left=np.min(contour[:,0])
    right=np.max(contour[:,0])
    centerY=int(np.mean([top,bottom]))
    centerX=int(np.mean([right,left]))
    new_contour=np.zeros_like(contour)
    for n,(x,y) in enumerate(contour):
        dy=y-centerY
        dx=x-centerX
        l=np.sqrt(dy*dy+dx*dx)
        deltay=pixel/l*dy
        deltax=pixel/l*dx
        newy=deltay+centerY
        newx=deltax+centerX
        new_contour[n,0]=int(newx)
        new_contour[n,1]=int(newy)
    if lencontour>2:
        new_contour=new_contour[:,None,:]
    return new_contour




# import numpy as np
# import itertools as IT
# import copy
# from shapely.geometry import LineString, Point

# def getPolyCenter(points):
#     """
#     http://stackoverflow.com/a/14115494/190597 (mgamba)
#     """
#     area = area_of_polygon(*zip(*points))
#     result_x = 0
#     result_y = 0
#     N = len(points)
#     points = IT.cycle(points)
#     x1, y1 = next(points)
#     for i in range(N):
#         x0, y0 = x1, y1
#         x1, y1 = next(points)
#         cross = (x0 * y1) - (x1 * y0)
#         result_x += (x0 + x1) * cross
#         result_y += (y0 + y1) * cross
#     result_x /= (area * 6.0)
#     result_y /= (area * 6.0)
#     return (result_x, result_y)

# def expandPoly(points, factor):
#     points = np.array(points, dtype=np.float64)
#     expandedPoly = points*factor
#     expandedPoly -= getPolyCenter(expandedPoly)
#     expandedPoly += getPolyCenter(points)
#     return np.array(expandedPoly, dtype=np.int64)

# def distanceLine2Point(points, point):
#     points = np.array(points, dtype=np.float64)
#     point = np.array(point, dtype=np.float64)

#     points = LineString(points)
#     point = Point(point)
#     return points.distance(point)

# def distancePolygon2Point(points, point):
#     distances = []
#     for i in range(len(points)):
#         if i==len(points)-1:
#             j = 0
#         else:
#             j = i+1
#         line = [points[i], points[j]]
#         distances.append(distanceLine2Point(line, point))

#     minDistance = np.min(distances)
#     #index = np.where(distances==minDistance)[0][0]

#     return minDistance 

# """
#     Returns the distance from a point to the nearest line of the polygon,
#     AND the distance from where the normal to the line (to reach the point)
#     intersets the line to the center of the polygon.
# """
# def distancePolygon2PointAndCenter(points, point):
#     distances = []
#     for i in range(len(points)):
#         if i==len(points)-1:
#             j = 0
#         else:
#             j = i+1
#         line = [points[i], points[j]]
#         distances.append(distanceLine2Point(line, point))

#     minDistance = np.min(distances)
#     i = np.where(distances==minDistance)[0][0]
#     if i==len(points)-1:
#         j = 0
#     else:
#         j = i+1
#     line = copy.deepcopy([points[i], points[j]])

#     centerDistance = distanceLine2Point(line, getPolyCenter(points))

#     return minDistance, centerDistance

