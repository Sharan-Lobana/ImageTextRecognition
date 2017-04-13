from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2
import random
import os
import pickle
import time
useSVM = False
useMLP = False
useCNN = True

USE_SMALL = True
C_VAL = 10.0
SHRINK_X = 0.5
SHRINK_Y = 0.5
TEST_SET_FRAC = 0.15
DATA_FRAC = .3
NUM_CLASSES = 36
mypath = './Train'

NFM = 16    #Number of feature maps
FMDIM = 11 #Feature map dimension

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
    # K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

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
            Y.append(i)
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

# Convert to numpy array and normalize if using CNN
if useCNN:
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

clf = None
if useSVM:
    # Don't use rbf kernel, gives bad testing error
    clf = svm.SVC(
        C=C_VAL,
        kernel='linear',
        verbose=True,
        probability=True,
        cache_size=200,
        class_weight=None,
        random_state=None,
        coef0=0.0,
        decision_function_shape='ovo',
        degree=3,
        gamma='auto',
        max_iter=-1,
        shrinking=True,
        tol=0.001,
    )
elif useMLP:
    clf = MLPClassifier(
        activation='tanh',
        solver='lbfgs',
        alpha=20,
        hidden_layer_sizes=(int(128*128*SHRINK_X*SHRINK_Y),int(64*64*SHRINK_X*SHRINK_Y)),
        random_state=1,
        learning_rate_init=0.1,
        max_iter=100,
        learning_rate='constant',
        verbose=True,
        tol=0.000001
    )
    # print clf.out_activation_
elif useCNN:
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
    # model.add(
    #     Conv2D(
    #         filters=4*NFM,
    #         kernel_size=(FMDIM,FMDIM),
    #         padding='valid',
    #         activation='relu',
    #         kernel_constraint=maxnorm(3)
    #     )
    # )
    # model.add(Dropout(0.2))
    # model.add(
    #     Conv2D(
    #         filters=8*NFM,
    #         kernel_size=(FMDIM,FMDIM),
    #         padding='valid',
    #         activation='relu',
    #         kernel_constraint=maxnorm(3)
    #     )
    # )
    # model.add(MaxPooling2D(pool_size=(2,2)))
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
        NUM_CLASSES,
        activation='softmax'
        )
    )

    epochs = 15
    lrate = 0.01
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
if not useCNN:
    clf.fit(x,y)

    resulttrain = clf.predict(x)
    resulttest = clf.predict(xtest)
    resultprob = clf.predict_proba(xtest)
    print resultprob
    # print "RESULTTEST"
    # print resulttest
    # print len(resulttest)
    # print "YTEST"
    # print ytest
    trainError = 0
    for i in range(len(resulttrain)):
        if resulttrain[i] != y[i]:
            trainError += 1

    testError = 0
    testErrorWOCase = 0
    for i in range(len(resulttest)):
        if resulttest[i] != ytest[i]:
            testError += 1
            testErrorWOCase += 1
            # Update test error without considering case
            if((resulttest[i] > 9) and (ytest[i] > 9)):
                if(abs(resulttest[i] - ytest[i]) == 26):
                    testErrorWOCase -= 1
            # print "Result is: "+str(resulttest[i]),
            # print " Desired is: "+str(ytest[i])

    trainErrorFrac = trainError/float(len(x))
    testErrorFrac = testError/float(len(xtest))
    testErrorWOCaseFrac = testErrorWOCase/float(len(xtest))
    print "Training error: "+str(trainErrorFrac)
    print "Testing error: "+str(testErrorFrac)
    print "Testing error w/o case : "+str(testErrorWOCaseFrac)

    # Save the trained SVM for further use
    prefix = None
    if useSVM:
        prefix = "MCSVC_"
    elif useMLP:
        prefix = "MLP_"
    elif useCNN:
        prefix="CNN_"
    pickle_filename = "./TrainedPickles/"+prefix+str(int(time.time()))[-6:]+"_"+str(NUM_CLASSES)+"_"+str(SHRINK_X)+"_"+\
    str(SHRINK_Y)+"_"+str(int((1-testErrorFrac)*10000)/100.0)+"_"+str(int((1-testErrorWOCaseFrac)*10000)/100.0)+".sav"
    pickle.dump(clf,open(pickle_filename,'wb'))
