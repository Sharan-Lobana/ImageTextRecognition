
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import numpy as np
import pickle
CNT_INNER_RECT = 2

def contains(rect1, rect2):
    if rect1[0][0]-rect2[0][0] > .33*min(rect1[1][0]-rect1[0][0], rect2[1][0]-rect2[0][0]):
        return False
    if rect1[0][1]-rect2[0][1] > .33*min(rect1[1][1]-rect1[0][1], rect2[1][1]-rect2[0][1]):
        return False
    if rect2[1][0]-rect1[1][0] > .33*min(rect1[1][0]-rect1[0][0], rect2[1][0]-rect2[0][0]):
        return False
    if rect2[1][1]-rect1[1][1] > .33*min(rect1[1][1]-rect1[0][1], rect2[1][1]-rect2[0][1]):
        return False
    return True

def decoded(x):
    if(x<=9):
        return str(x)
    elif(x<=35):
        return chr(ord('a')+x-10)
    elif(x<=61):
        return chr(ord('a')+x-36)
    else:
        return "What?"

def area(rect):
    return (rect[1][0]-rect[0][0])*(rect[1][1]-rect[0][1])

def compare(item1, item2):
    if area(item1) < area(item2):
        return -1
    elif area(item1) > area(item2):
        return 1
    else:
        return 0

def get_probable_rects(img, remove_redundant = True, rects = []):
    mser = cv2.MSER()
    regions = mser.detect(img, None)    #detect and extract MSER lasso-contours
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  #take convex-hull of each contour

    temp = img.copy()
    cv2.polylines(temp, hulls, 1, (255, 255, 255))
    cv2.imshow('MSER Convex Hulls', temp)
    cv2.waitKey(0)

    temp = img.copy()

    for hull in hulls:
        x,y,w,h = cv2.boundingRect(hull)    #convert the hull to bounding rectangle
        cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),1)
        rects.append([[x,y],[x+w,y+h]])     #add it to list for processing
    cv2.imshow('Bounding Boxes', temp)
    cv2.waitKey(0)
        
    rects = sorted(rects, cmp=compare)
    
    if not remove_redundant:
        return rects

    ITER = 0
    mark= []
    for i in range(len(rects)):
        mark.append(False)
    for i in range(len(rects)):
        # print i, "of ", len(rects), "iter: ", ITER
        if mark[i]:
            continue
        Count = 0
        for j in range(len(rects)):
            if i==j:
                continue
            if mark[j]:
                continue
            if contains(rects[i], rects[j]):
                Count += 1
        if Count > 1:   #if a rectangle contains more than 1 rectangle, remove it
            mark[i] = True
            continue
        for j in range(len(rects)):
            if i==j:
                continue
            if mark[j]:
                continue
            if contains(rects[i], rects[j]):    #else remove the inner rectangle
                mark[j] = True
    temp_rects = []
    for i in range(len(rects)):
        if not mark[i]:
            temp_rects.append(rects[i])
    return temp_rects

img = cv2.imread('sink.jpg',0)  #read black and white image
vis = img.copy()
rects = get_probable_rects(vis)

height, width = img.shape
cv2.rectangle(img,(0,0),(width-1,height-1),(100,100,100),-1)

for rect in rects:
    # cv2.rectangle(mask,(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(255,255,255),-1)  #plot the ractangles on image
    dx = rect[1][0]-rect[0][0]
    dy = rect[1][1]-rect[0][1]
    letter = vis[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
    # thresh,letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print thresh
    img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]] = letter

temp = img.copy()
cv2.imshow('Removed Redundant MSERs', temp)
cv2.waitKey(0)

for rect in rects:
    # cv2.rectangle(mask,(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(255,255,255),-1)  #plot the ractangles on image
    dx = rect[1][0]-rect[0][0]
    dy = rect[1][1]-rect[0][1]
    letter = vis[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
    thresh,letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print thresh
    img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]] = letter

temp = img.copy()
cv2.imshow('Binarized MSERs', temp)
cv2.waitKey(0)

cv2.imwrite('plot_rects_binarized_1.jpg', img)
img = cv2.imread('plot_rects_binarized_1.jpg',0)

# print len(rects)
# for rect in rects:
#     dx = rect[1][0]-rect[0][0]
#     dy = rect[1][1]-rect[0][1]
#     Sum = 0
#     cnt = 0
#     for i in range(dx):
#         Sum += img[max(rect[0][1]-5,0),rect[0][0]+i]
#         Sum += img[min(rect[1][1]+5,height-1),rect[0][0]+i]
#         cnt += 2
#     for i in range(dy):
#         Sum += img[rect[0][1]+i,max(rect[0][0]-5,0)]
#         Sum += img[rect[0][1]+i,min(rect[1][0]+5,width-1)]
#         cnt += 2
#     Sum = Sum/float(cnt)
#     if Sum < 120:
#         for i in range(dy):
#             for j in range(dx):
#                 img[rect[0][1]+i,rect[0][0]+j] = 255 - img[rect[0][1]+i,rect[0][0]+j]
#     cv2.rectangle(img,(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(175,175,175),1)

# # vis = cv2.bitwise_and(vis,mask)
# # cv2.polylines(vis, hulls, 1, (255, 255, 255))

# # vis = cv2.adaptiveThreshold(vis,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,19,0)


# cv2.imshow('img', img)
# cv2.imwrite('plot_rects_binarized_1.jpg', img)

cv2.destroyAllWindows()
