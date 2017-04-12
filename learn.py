from sklearn import svm
import numpy as np
import cv2
import random
import os
import pickle
import time

USE_SMALL = True
C_VAL = 5.0
SHRINK_X = 0.25
SHRINK_Y = 0.25
TEST_SET_FRAC = 0.15
DATA_FRAC = .3
NUM_CLASSES = 62
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
        img = cv2.imread(path+os.sep+cf,0)
        if(USE_SMALL):
            small = cv2.resize(img, (0,0), fx=SHRINK_X, fy=SHRINK_Y)
            X.append(list(small.flatten()))
        else:
            X.append(list(img.flatten()))
        Y.append(i)

print len(X)
print len(Y)

xtest = []
ytest = []
x = []
y = []

temp = 0
for ind in range(NUM_CLASSES):
    indices = list(np.random.permutation(1016))
    indices = [i+(ind*1016) for i in indices]
    for i in indices:
        if random.random() < DATA_FRAC:
            if random.random() < TEST_SET_FRAC:
                xtest.append(X[i])
                ytest.append(Y[i])
            else:
                x.append(X[i])
                y.append(Y[i])
    if(ind <= 9):
        print "Number of test instances for "+str(ind)+" "+str(len(xtest)-temp)
    elif(ind <= 35):
        print "Number of test instances for "+chr(ord('a')+ind-10)+" "+str(len(xtest)-temp)
    else:
        print "Number of test instances for "+chr(ord('A')+ind-36)+" "+str(len(xtest)-temp)
    temp = len(xtest)

# print "y is: ",
# print y
# Don't use rbf kernel, gives bad testing error
clf = svm.SVC(C=C_VAL,kernel='linear',verbose=True)
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
        # Update test error without considering case
        if((resulttest[i] > 9) and (ytest[i] > 9)):
            if(abs(resulttest[i] - ytest[i]) != 26):
                testErrorWOCase += 1
        print "Result is: "+str(resulttest[i]),
        print " Desired is: "+str(ytest[i])

trainErrorFrac = trainError/float(len(x))
testErrorFrac = testError/float(len(xtest))
testErrorWOCaseFrac = testErrorWOCase/float(len(xtest))
print "Training error: "+str(trainErrorFrac)
print "Testing error: "+str(testErrorFrac)
print "Testing error w/o case : "+str(testErrorWOCaseFrac)

# Save the trained SVM for further use
pickle_filename = "./SVMPickles/MCSVC_"+str(int(time.time()))+"_"+str(testErrorFrac)+".sav"
pickle.dump(clf,open(pickle_filename,'wb'))
