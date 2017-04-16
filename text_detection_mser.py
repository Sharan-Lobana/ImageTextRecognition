
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import cPickle, random
CNT_INNER_RECT = 2

useSVM = False
useMLP = False
useCNN = True
useTextNoTextClassifier = True
def generateRandomArray(s):
    arr = np.zeros(s)
    for i in range(8):
        for j in range(8):
            val = random.random()
            if val < .33:
                continue
            if val < .66:
                for x in range(s[0]/8):
                    for y in range(s[1]/8):
                        arr[i*(s[0]/8)+x][j*(s[1]/8)+y] = 255
            else:
                for x in range(s[0]/8):
                    for y in range(s[1]/8):
                        if random.random() < .5:
                            arr[i*(s[0]/8)+x][j*(s[1]/8)+y] = 255
    # cv2.imshow('Removed Redundant MSERs', arr)
    # cv2.waitKey(0)
    return arr

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

def mynormalization(image,new_dim,inverted=False):
    if inverted:
        myshape = image.shape
        for i in range(myshape[0]):
            for j in range(myshape[1]):
                image[i][j] = 255 - image[i][j]
    array_shape = image.shape
    min_dim = min(array_shape[0],array_shape[1])
    new_array = None
    if array_shape[0] != array_shape[1]:
        if array_shape[0] == min_dim:
            new_array = np.zeros((array_shape[1],array_shape[1]))
            new_array.fill(255)
            diff = array_shape[1] - array_shape[0]
            new_array[diff/2:array_shape[0]+diff/2,:] = image
            new_array = cv2.resize(new_array,(new_dim[0]-12,new_dim[1]-12))
        else:
            new_array = np.zeros((array_shape[0],array_shape[0]))
            new_array.fill(255)
            diff = array_shape[0] - array_shape[1]
            new_array[:,diff/2:array_shape[1]+diff/2] = image
            new_array = cv2.resize(new_array,(new_dim[0]-12,new_dim[1]-12))
    else:
        new_array = cv2.resize(image,(new_dim[0]-12,new_dim[1]-12))
    rarray = np.zeros((64,64))
    rarray.fill(255)
    rarray[6:-6,6:-6] = new_array
    return rarray

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
        return chr(ord('A')+x-36)
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

    mark= []
    for i in range(len(rects)):
        mark.append(False)
    for i in range(len(rects)):
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

generateRandomArray((64,64))
img = cv2.imread('test2.jpg',0)  #read black and white image
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
    #No. of columns = length, No. of rows = height

temp = img.copy()
cv2.imshow('Removed Redundant MSERs', temp)
cv2.waitKey(0)

X = []
for rect in rects:
    # cv2.rectangle(mask,(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(255,255,255),-1)  #plot the ractangles on image
    dx = rect[1][0]-rect[0][0]
    dy = rect[1][1]-rect[0][1]
    letter = vis[max(0,rect[0][1]-4):min(height-1,rect[1][1]+4),max(0,rect[0][0]-4):min(width-1,rect[1][0]+4)]
    thresh,letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print thresh
    img[max(0,rect[0][1]-4):min(height-1,rect[1][1]+4),max(0,rect[0][0]-4):min(width-1,rect[1][0]+4)] = letter

temp = img.copy()
cv2.imshow('Binarized MSERs', temp)
cv2.waitKey(0)

mark = img.copy()
for i in range(height):
    for j in range(width):
        mark[i,j]=0

# print len(rects)
for rect in rects:
    dx = rect[1][0]-rect[0][0]
    dy = rect[1][1]-rect[0][1]
    Sum = 0
    cnt = 0
    for i in range(dx):
        Sum += img[max(rect[0][1]-3,0),rect[0][0]+i]
        Sum += img[min(rect[1][1]+3,height-1),rect[0][0]+i]
        cnt += 2
    for i in range(dy):
        Sum += img[rect[0][1]+i,max(rect[0][0]-3,0)]
        Sum += img[rect[0][1]+i,min(rect[1][0]+3,width-1)]
        cnt += 2
    Sum = Sum/float(cnt)
    if Sum < 123:
        for i in range(dy):
            for j in range(dx):
                if mark[rect[0][1]+i,rect[0][0]+j]:
                    continue
                mark[rect[0][1]+i,rect[0][0]+j] = 1
                img[rect[0][1]+i,rect[0][0]+j] = 255 - img[rect[0][1]+i,rect[0][0]+j]

    char_array = img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
    X.append(char_array)
temp = img.copy()
cv2.imshow('Inverted Binarized MSERs', temp)
cv2.waitKey(0)
cv2.imwrite('plot_rects_binarized_1.jpg', img)
img = cv2.imread('plot_rects_binarized_1.jpg',0)

cv2.destroyAllWindows()
xlen = len(X)
for i in range(xlen):
    cv2.imwrite("./ExtrImgsBeforeNorm/"+str(i)+".jpg",X[i])

Xnorm = [mynormalization(image,(64,64)) for image in X]
Xnorm = np.asarray(Xnorm)

# Add an extra dimension for giving input to CNN
Xnorm = np.expand_dims(Xnorm,axis=3)
Ynorm = []

# ==============================================================================
# Using Text No Text classfication
text_not_text = None
if useTextNoTextClassifier:
    model_filename = "./TextNoTextModels/CNN_356922_621_0.5_0.50.98.json"
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print loaded_model.summary()
    # load weights into new model
    loaded_model.load_weights(model_filename[:-4]+"h5")
    print("Loaded model from disk")

    # predict text or not text
    text_not_text = loaded_model.predict_classes(Xnorm)
    print "####################################################################"
    print text_not_text
    print "####################################################################"

# ==============================================================================


for i in range(xlen):
    cv2.imwrite("./ExtrImgs/"+str(i)+".jpg",Xnorm[i])

if useSVM or useMLP:
    clf = pickle.load(open('./TrainedPickles/MLP_238174_3_0.5_0.5_95.85_95.85.sav','rb'))
    Xnorm = [image.flatten() for image in Xnorm]
    Ynorm = clf.predict_proba(Xnorm)
    print Ynorm

elif useCNN:
    model_filename = "./TrainedPickles/CNN_218692_62_0.5_0.50.91.json"
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print loaded_model.summary()
    # load weights into new model
    loaded_model.load_weights(model_filename[:-4]+"h5")
    print("Loaded model from disk")
    Ynorm = loaded_model.predict_proba(Xnorm)
    Y2norm = loaded_model.predict_classes(Xnorm)
    print Y2norm
    Y3 = [decoded(y) for y in Y2norm]
    print Y3
    temp = vis.copy()
    for i in range(len(Y3)):
        c = Y3[i]
        if text_not_text[i] == 1:
            cv2.rectangle(temp,(rects[i][0][0],rects[i][0][1]),(rects[i][1][0],rects[i][1][1]),255,2)
            cv2.putText(temp,c, (rects[i][0][0],rects[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, 0,1)
    cv2.imshow('prediction', temp)
    cv2.imwrite('result.jpg',temp)
    cv2.waitKey(0)
