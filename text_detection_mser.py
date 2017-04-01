
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
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

def area(rect):
    return (rect[1][0]-rect[0][0])*(rect[1][1]-rect[0][1])

def compare(item1, item2):
    if area(item1) < area(item2):
        return -1
    elif area(item1) > area(item2):
        return 1
    else:
        return 0

img = cv2.imread('test.jpg',0);  #read black and white image
vis = img.copy()
mser = cv2.MSER()
regions = mser.detect(img, None)    #detect and extract MSER lasso-contours
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  #take convex-hull of each contour

rects = []

for hull in hulls:
    x,y,w,h = cv2.boundingRect(hull)    #convert the hull to bounding rectangle
    rects.append([[x,y],[x+w,y+h]])     #add it to list for processing
    # cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1)

rects = sorted(rects, cmp=compare)  #Sort the bounding rectangles in asc order


ITER = 0

# Heuristic to remove a rectangle if it contains more than
# one inner rectangles. If a rectangle is surrounded by a
# larger rectangle, the inner one is also removed
while True:
    ITER += 1
    change = False
    print ITER
    for i in range(len(rects)):
        # print i, "of ", len(rects), "iter: ", ITER
        Count = 0
        for j in range(len(rects)):
            if i==j:
                continue
            if contains(rects[i], rects[j]):
                Count += 1
        if Count > CNT_INNER_RECT:   #if a rectangle contains more than 1 rectangle, remove it
            temp_rects = []
            for k in range(len(rects)):
                if k==i:
                    continue
                temp_rects.append(rects[k])
            rects = temp_rects
            change = True
            break

        for j in range(len(rects)):
            if i==j:
                continue
            if contains(rects[i], rects[j]):    #else remove the inner rectangle
                temp_rects = []
                for k in range(len(rects)):
                    if j==k:
                        continue
                    temp_rects.append(rects[k])
                rects = temp_rects
                change = True
                break
        if change:
            break

    if not change:
        break

for rect in rects:
    cv2.rectangle(vis,(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(0,255,0),1)  #plot the ractangles on image

# cv2.polylines(vis, hulls, 1, (0, 255, 0))
x1 = [i[0][0] for i in rects]
y1 = [-i[0][1] for i in rects]

plt.scatter(x1,y1)
plt.show()


print len(rects)

# cv2.imshow('img', vis)
# cv2.imwrite('union.jpg', vis)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
