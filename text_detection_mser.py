
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import numpy as np
import pickle
CNT_INNER_RECT = 2

useSVM = False
useMLP = False
useCNN = True
if useCNN:
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.constraints import maxnorm
    from keras.optimizers import SGD
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    from keras.models import model_from_json

seed = 7
np.random.seed(seed)

def mynormalization(image,new_dim):
    array_shape = image.shape
    min_dim = min(array_shape[0],array_shape[1])
    new_array = None
    if array_shape[0] != array_shape[1]:
        if array_shape[0] == min_dim:
            new_array = np.zeros((array_shape[1],array_shape[1]))
            diff = array_shape[1] - array_shape[0]
            new_array[diff/2:array_shape[0]+diff/2,:] = image
            new_array = cv2.resize(new_array,(new_dim[0],new_dim[1]))
        else:
            new_array = np.zeros((array_shape[0],array_shape[0]))
            diff = array_shape[0] - array_shape[1]
            new_array[:,diff/2:array_shape[1]+diff/2] = image
            new_array = cv2.resize(new_array,(new_dim[0],new_dim[1]))
    else:
        new_array = cv2.resize(image,(new_dim[0],new_dim[1]))
    return new_array

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

X = []
for rect in rects:
    length = rect[1][0]-rect[0][0]+1
    height = rect[1][1]-rect[0][1]+1
    print "X: "+str(rect[0][0])+" Y: "+str(rect[0][1]),
    print " Length: "+str(length)+" Height: "+str(height)
    #No. of columns = length, No. of rows = height
    char_array = img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
    X.append(char_array)
# cv2.polylines(vis, hulls, 1, (0, 255, 0))

X = [mynormalization(image,(64,64)) for image in X]
X = np.asarray(X)
X = np.expand_dims(X,axis=3)
Y = []
if useSVM or useMLP:
    X = [image.flatten() for image in X]
    clf = pickle.load(open('./SVMPickles/MCSVC_1491072043_0.0496169281284.sav','rb'))
    Y = clf.predict(X)
elif useCNN:
    model_filename = "./TrainedPickles/CNN_218692_62_0.5_0.50.91.json"
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_filename[:-4]+"h5")
    print("Loaded model from disk")
    Y = loaded_model.predict_proba(X)
    Y2 = loaded_model.predict_classes(X)
    print Y
    print Y2
    Y3 = [decoded(y) for y in Y2]
    print Y3

# Y = [decoded(y) for y in Y]
# print Y

x1 = [i[0][0] for i in rects]
y1 = [-i[0][1] for i in rects]

plt.scatter(x1,y1)
plt.show()

print len(rects)
