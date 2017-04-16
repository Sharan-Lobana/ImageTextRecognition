from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2
import random
import os
import pickle
import time
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

seed = 7
np.random.seed(seed)

SHRINK_X = 0.5
SHRINK_Y = 0.5
TEST_SET_FRAC = 0.15
DATA_FRAC = 0.03
DATA_FRAC2 = 1.0
NUM_CLASSES = 62
NUM_CLASSES2 = 1
mypath = './Train'
mypath2 = './NonTextTrain'
useCNN = True
USE_SMALL = True
RESIZE_NON_TEXT = True

NFM = 16    #Number of feature maps
FMDIM = 11 #Feature map dimension

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

    return arr
directories = []
for (dirpath, dirnames, filenames) in os.walk(mypath):
    directories.extend(dirnames)
    break

print len(directories)
directories.sort()
X = []
Y = []
directory = ''
count_array = [0]*NUM_CLASSES
count = 0
for i in range(len(directories)):
    if(i >= NUM_CLASSES):
        break
    curr_files = []
    directory = directories[i]
    path = mypath + os.sep + directory
    for (dirpath, dirnames, filenames) in os.walk(path):
        curr_files.extend(filenames)
        break
    curr_files.sort()
    count += 1
    print "Curr_files "+str(count)+" length: "+str(len(curr_files))
    for cf in curr_files:
        if random.random() < DATA_FRAC:
            img = cv2.imread(path+os.sep+cf,0)
            if(USE_SMALL):
                small = cv2.resize(img, (0,0), fx=SHRINK_X, fy=SHRINK_Y)
                if(useCNN):
                    X.append(small)
                else:
                    X.append(list(small.flatten()))
            else:
                if(useCNN):
                    X.append(img)
                else:
                    X.append(list(img.flatten()))
            Y.append(1) # 1 for text
            count_array[i] += 1

print len(X)
print len(Y)

xtest = []
ytest = []
x = []
y = []

temp = 0
for ind in range(NUM_CLASSES):
    indices = list(np.random.permutation(count_array[ind]))
    indices = [i+sum(count_array[0:ind]) for i in indices]
    for i in indices:
        if random.random() < TEST_SET_FRAC:
            xtest.append(X[i])
            ytest.append(Y[i])
        else:
            x.append(X[i])
            y.append(Y[i])
    if(ind <= 9):
        print "Number of train instances for "+str(ind)+" "+str(count_array[ind]-len(xtest)+temp)
        print "Number of test instances for "+str(ind)+" "+str(len(xtest)-temp)
    elif(ind <= 35):
        print "Number of train instances for "+chr(ord('a')+ind-10)+" "+str(count_array[ind]-len(xtest)+temp)
        print "Number of test instances for "+chr(ord('a')+ind-10)+" "+str(len(xtest)-temp)
    else:
        print "Number of train instances for "+chr(ord('A')+ind-36)+" "+str(count_array[ind]-len(xtest)+temp)
        print "Number of test instances for "+chr(ord('A')+ind-36)+" "+str(len(xtest)-temp)
    temp = len(xtest)

print "========================================================================"
print "========================================================================"
print "========================================================================"
print "========================================================================"
print "========================================================================"
print "========================================================================"

directories2 = []
for (dirpath, dirnames, filenames) in os.walk(mypath2):
    directories2.extend(dirnames)
    break

print len(directories2)
directories2.sort()
X2 = []
Y2 = []
directory2 = ''
count_array2 = [0]*NUM_CLASSES2
count2 = 0
for i in range(len(directories2)):
    if(i >= NUM_CLASSES2):
        break
    curr_files2 = []
    directory2 = directories2[i]
    path = mypath2 + os.sep + directory2
    for (dirpath, dirnames, filenames) in os.walk(path):
        curr_files2.extend(filenames)
        break
    curr_files2.sort()
    count2 += 1
    print "Curr_files2 "+str(count2)+" length: "+str(len(curr_files2))
    for cf in curr_files2:
        if random.random() < DATA_FRAC2:
            img = cv2.imread(path+os.sep+cf,0)
            if(RESIZE_NON_TEXT):
                temparray = img.copy()
                # print img.shape
                if(img.shape[0] != 52 or img.shape[1] != 52):
                    continue
                temparray = cv2.resize(temparray,(64,64))
                cv2.rectangle(temparray,(0,0),(63,63),255,-1)
                small = img.copy()
                temparray[6:-6,6:-6] = small
                small = temparray
                # print small.type()
                thresh,small = cv2.threshold(small,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                if useCNN:
                    X2.append(small)
                else:
                    X2.append(list(small.flatten()))
            else:
                if(useCNN):
                    thresh,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    X2.append(img)
                else:
                    X2.append(list(img.flatten()))
            Y2.append(0)    # 0 For non-text
            count_array2[i] += 1

# # Adding random noise for non-text training
# for i in range(20000):
#     X2.append(generateRandomArray((int(128*SHRINK_X),int(128*SHRINK_Y))))
#     Y2.append(0) # 0 For non-text


print "X is : "+str(len(X))
print "Y is : "+str(len(Y))
print "X2 is : "+str(len(X2))
print "Y2 is : "+str(len(Y2))

xtest2 = []
ytest2 = []
x2 = []
y2 = []

temp2 = 0
for ind in range(NUM_CLASSES2):
    indices = list(np.random.permutation(count_array2[ind]))
    indices = [i+sum(count_array2[0:ind]) for i in indices]
    for i in indices:
        if random.random() < TEST_SET_FRAC:
            xtest2.append(X2[i])
            ytest2.append(Y2[i])
        else:
            x2.append(X2[i])
            y2.append(Y2[i])
    if(ind <= 9):
        print "Number of train instances for image number "+str(ind)+" "+str(count_array2[ind]-len(xtest2)+temp2)
        print "Number of test instances for image number "+str(ind)+" "+str(len(xtest2)-temp2)
    temp2 = len(xtest2)

# TODO: write code for merging x,x2 and y,y2 and xtest,xtest2 and ytest,ytest2


x.extend(x2)
xtest.extend(xtest2)

y.extend(y2)
ytest.extend(ytest2)
mylist = random.sample(range(0,len(x)),len(x))

x_train = []
y_train = []
x_test = []
y_test= []
for i in range(len(mylist)):
    x_train.append(x[mylist[i]])
    y_train.append(y[mylist[i]])

mylist = random.sample(range(0,len(xtest)),len(xtest))
for i in range(len(mylist)):
    x_test.append(xtest[mylist[i]])
    y_test.append(ytest[mylist[i]])

x = x_train
y = y_train
xtest = x_test
ytest = y_test

# Convert to numpy array and normalize if using CNN

x = np.asarray(x).astype('float32')
xtest = np.asarray(xtest).astype('float32')

y = np.asarray(y)
ytest = np.asarray(ytest)
# Normalize Inputs
x = x/255.0
xtest = xtest/255.0

# add an extra dimension for convolution
x = np.expand_dims(x,axis=3)
xtest = np.expand_dims(xtest,axis=3)
# one hot encode outputs (conversion to binary matrix)
y = np_utils.to_categorical(y)
ytest = np_utils.to_categorical(ytest)

print "NUM_CLASSES: "+str(ytest.shape[1])


model = Sequential()
# Add a convolutional 2D layer with number of feature maps equal to NFM,
# dimension of feature map equal to FMDIM x FMDIM, shape of the input as
# specified , activation as specified and a constraint that maxnorm of any
# hidden layer weight vector will be 3
model.add(
    Conv2D(
        filters=NFM,
        kernel_size=(FMDIM,FMDIM),
        input_shape=(int(128*SHRINK_X),int(128*SHRINK_Y),1),
        padding='same',
        activation='relu',
        kernel_constraint=maxnorm(3)
    )
)
model.add(Dropout(0.2))
model.add(
    Conv2D(
        filters=2*NFM,
        kernel_size=(FMDIM,FMDIM),
        padding='valid',
        activation='relu',
        kernel_constraint=maxnorm(3)
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(
    Conv2D(
        filters=4*NFM,
        kernel_size=(FMDIM,FMDIM),
        padding='valid',
        activation='relu',
        kernel_constraint=maxnorm(3)
    )
)
model.add(Dropout(0.2))
model.add(
    Conv2D(
        filters=8*NFM,
        kernel_size=(FMDIM,FMDIM),
        padding='valid',
        activation='relu',
        kernel_constraint=maxnorm(3)
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(
    Dense(
        256,
        activation='relu',
        kernel_constraint=maxnorm(3)
    )
)
model.add(Dropout(0.5))
model.add(
    Dense(
    2,
    activation='softmax'
    )
)

epochs = 5
lrate = 0.001
decay = lrate/epochs
sgd = SGD(
    lr=lrate,
    momentum=0.9,
    decay=decay,
    nesterov=False
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

print(model.summary())

model.fit(
    x,
    y,
    validation_data=(xtest,ytest),
    epochs=epochs,
    batch_size=32
)
scores = model.evaluate(xtest, ytest, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


prefix="CNN_"
pickle_filename = "./TextNoTextModels/"+prefix+str(int(time.time()))[-6:]+"_"+\
str(NUM_CLASSES)+str(NUM_CLASSES2)+"_"+str(SHRINK_X)+"_"+str(SHRINK_Y)
model_json = model.to_json()
pickle_filename = pickle_filename+"%.2f"%scores[1]+".json"
with open(pickle_filename, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(pickle_filename[:-4]+"h5")
print("Saved model to disk")
