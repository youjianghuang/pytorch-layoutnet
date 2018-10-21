# coding: utf-8

'''
@author: eju
'''

import numpy as np
import PIL
import sys
import cv2

sys.path.insert(1, 'D:\\program\\pytorch-layoutnet')
import pano_lsd_align

cutSize = 320
fov = np.pi / 3

xh = np.arange(-np.pi, np.pi*5/6, np.pi/6)
yh = np.zeros(xh.shape[0])
xp = np.array([-3/3, -2/3, -1/3, 0/3,  1/3, 2/3, -3/3, -2/3, -1/3,  0/3,  1/3,  2/3]) * np.pi
yp = np.array([ 1/4,  1/4,  1/4, 1/4,  1/4, 1/4, -1/4, -1/4, -1/4, -1/4, -1/4, -1/4]) * np.pi
x = np.concatenate([xh, xp, [0, 0]])
y = np.concatenate([yh, yp, [np.pi/2., -np.pi/2]])

im = PIL.Image.open('pano.jpg')

im_array = np.array(im)

sepScene = pano_lsd_align.separatePano(im_array.copy(), fov, x, y, cutSize)
 
print(len(sepScene))

scene_16 = sepScene[16]
PIL.Image.fromarray(scene_16['img'].astype(np.uint8)).save('separate_0_origin.png')

edge = []
LSD = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV, _quant=0.7)
gray_img = cv2.cvtColor(scene_16['img'], cv2.COLOR_RGB2GRAY)
PIL.Image.fromarray(gray_img.astype(np.uint8)).save('separate_0_gray.png')

lines, width, prec, nfa = LSD.detect(gray_img)

edgeMap = LSD.drawSegments(np.zeros_like(gray_img), lines)[..., -1]

PIL.Image.fromarray(edgeMap.astype(np.uint8)).save('separate_0_edge.png')

print(lines.shape)
lines = np.squeeze(lines, 1)    # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
print(lines.shape)
edgeList = np.concatenate([lines, width, prec, nfa], 1)

#print(edgeList)
#print(edgeList.shape)

edge = {
    'img': edgeMap,
    'edgeLst': edgeList,
    'vx': scene_16['vx'],
    'vy': scene_16['vy'],
    'fov': scene_16['fov']
}

print(edge['edgeLst'].shape)

# 计算panoLst
edgeList = edge['edgeLst']
vx = edge['vx']
vy = edge['vy']
fov = edge['fov']
imH, imW = edge['img'].shape
print(imH,imW)

R = (imW/2) / np.tan(fov/2)
print("R:", R)

# im is the tangent plane, contacting with ball at [x0 y0 z0]
x0 = R * np.cos(vy) * np.sin(vx)
y0 = R * np.cos(vy) * np.cos(vx)
z0 = R * np.sin(vy)
print("x0,y0,z0: ", x0, y0, z0)
vecposX = np.array([np.cos(vx), -np.sin(vx), 0])
vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
vecposY = vecposY / np.sqrt(vecposY @ vecposY.T)
vecposX = vecposX.reshape(1, -1)
vecposY = vecposY.reshape(1, -1)
Xc = (0 + imW-1) / 2
Yc = (0 + imH-1) / 2

#print("Xc,Yc: ", Xc, Yc)

vecx1 = edgeList[:, [0]] - Xc
vecy1 = edgeList[:, [1]] - Yc
vecx2 = edgeList[:, [2]] - Xc
vecy2 = edgeList[:, [3]] - Yc

print("vecPosX VecPosY: ", vecposX, vecposY)
print("Xc YC: ", Xc, Yc)
print("vecx1 vecy1 vecx2 vecy2: ", vecx1[0], vecy1[0], vecx2[0], vecy2[0])

vec1 = np.tile(vecx1, [1, 3]) * vecposX + np.tile(vecy1, [1, 3]) * vecposY
vec2 = np.tile(vecx2, [1, 3]) * vecposX + np.tile(vecy2, [1, 3]) * vecposY

print("vec1 vec2: ", vec1[0], vec2[0])

coord1 = [[x0, y0, z0]] + vec1
coord2 = [[x0, y0, z0]] + vec2

normal = np.cross(coord1, coord2, axis=1)
normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

panoList = np.hstack([normal, coord1, coord2, edgeList[:, [-1]]])

# print(vx,vy,fov)
# 
# 
# print(edgeList[0])
# print("normal: ", normal[0])
# print("coord1: ", coord1[0])
# print("coord2: ", coord2[0])
# print(edgeList[:, [-1]][0])

# return panoList


# edge['panoLst'] = pano_lsd_align.edgeFromImg2Pano(edge)
# 
# print(edge['panoLst'].shape)

# edgeLst
# [x1-------------y1-------------x2-------------y2-------------width----------prec-----------nfa-----------]
# [8.97189808e+00 1.40631226e+02 7.87251377e+00 6.93788986e+01 3.23251407e+00 1.25000000e-01 4.11825416e+02]
# print(edge['edgeLst'][0])
# panoList
# [-8.79043790e-01 -4.76548062e-01  1.35631633e-02  1.50528102e+02 -2.77128129e+02  1.88687744e+01  1.51627486e+02 -2.77128129e+02 9.01211014e+01  4.11825416e+02]
# print(edge['panoLst'][0])

# print(edge['vx'], edge['vy'], edge['fov'])
