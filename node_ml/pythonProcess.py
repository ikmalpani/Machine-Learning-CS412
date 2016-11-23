import numpy as np
import json

file = open('allTime.csv', 'r')

all = []
for line in file:
	if "head" not in line and len(line)> 1:
		tokens = line.split(",")
		if( len(tokens) > 71 ):
			all.append(line)

np.random.shuffle(all)

# subsets = []
# for i in range(0, 10):
# 	s = []
# 	subsets.append(s)

# for i in range(0, len(all) )
# 	idx = (int) i / 10
# 	subsets[idx].append(all[i])
lenAll = len(all)
stopPoint = len(all)*.9

allTest = []
allTrain = []

countSubjects = 0
countSeventy = 0
countOneFortyThree = 0
for i in range(0, len(all)):
	# if "head" in line:
	# 	countSubjects = countSubjects + 1
	# elif len(line)>1:
	line = all[i] 
	tokens = line.split(',')
	whichState = tokens[2]
	stateInt = 1

	if "free" in whichState or "init" in whichState:
		stateInt = -1
	row = []

	whichPrior = tokens[1]
	priorInt = 1
	if "free" in whichState or "init" in whichState:
		priorInt = -1
	
	row.append(priorInt)

	for j in range(3, len(tokens)):
		row.append( float(tokens[j]))

	rowAndClass = []
	rowAndClass.append(row)
	rowAndClass.append(stateInt)
	if( len(row) > 71 ):
		if i < stopPoint:
			allTrain.append(rowAndClass)
		else:
			allTest.append(rowAndClass)

	# print len(row)
	# if( len(row) == 70 ):
	# 	countSeventy = countSeventy +1
	# 	print countSeventy
	# else:
	# 	countOneFortyThree = countOneFortyThree + 1
	# 	print countOneFortyThree

# testfile = open("test.csv", w)
# trainfile = open("train.csv", w)

# str = ""
print len(allTest)

trainFile = open("train.json", 'w')
testFile = open("test.json", 'w')

allTr = []
for i in range(0, len(allTrain)):
	allTr.append(allTrain[i])
json.dump(allTr, trainFile)
# pickle.dump(allTr, trainFile)

allTe = []
for i in range(0, len(allTest)):
	allTe.append(allTest[i])
json.dump(allTe, testFile)

trainFile.close()
testFile.close()
# pickle.dump(allTe, testFile)

# print everything
	# for j in range
	# # str = "[ " + allTrain[i] + "]," + allTrain[i+1]
	# # print str
	# # i = i + 1 