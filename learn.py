from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2
import random
import os
import pickle
import time
useSVM = False
useNN = True
USE_SMALL = True
C_VAL = 10.0
SHRINK_X = 0.25
SHRINK_Y = 0.25
TEST_SET_FRAC = 0.15
DATA_FRAC = .9
NUM_CLASSES = 10
mypath = './Train'
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
                X.append(list(small.flatten()))
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

# print "y is: ",
# print y
# Don't use rbf kernel, gives bad testing error
clf = None
if useSVM:
    clf = svm.SVC(
        C=C_VAL,
        kernel='linear',
        verbose=True
    )
elif useNN:
    clf = MLPClassifier(
        solver='lbfgs',
        alpha=100,
        hidden_layer_sizes=(int(32*32), NUM_CLASSES),
        random_state=1,
        learning_rate_init=0.01,
        max_iter=1000,
        learning_rate='constant',
        verbose=True,
    )

clf.fit(x,y)

resulttrain = clf.predict(x)
resulttest = clf.predict(xtest)
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
        print "Result is: "+str(resulttest[i]),
        print " Desired is: "+str(ytest[i])

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
elif useNN:
    prefix = "NN_"
pickle_filename = "./TrainedPickles/"+prefix+str(int(time.time()))[-6:]+"_"+str(NUM_CLASSES)+\
"_"+str(int((1-testErrorFrac)*10000)/100.0)+"_"+str(int((1-testErrorWOCaseFrac)*10000)/100.0)+".sav"
pickle.dump(clf,open(pickle_filename,'wb'))
