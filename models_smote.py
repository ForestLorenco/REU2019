from __future__ import absolute_import, division, print_function, unicode_literals

#this is for oversampling, using smote
import pandas as pd
import collections
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

#sklearn for SVM
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


from sklearn.svm import SVC

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm
from tqdm import trange

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import seaborn as sns

#This does the smote process for oversampling
def doSMOTE(filename):
	print("SMOTE on "+filename)
	#Load in the csv of the data from NNData
	df = pd.read_csv("NN_SVMData/"+filename)
	print("========Head of the Data========")
	print(df.head())
	#this shows there is a harsh imbalance of the classes (positive, negative, neutral)
	print("========Count of Classes========")
	print(df['class'].value_counts())

	#This is to convert the classes to numbers
	#print("========Head of the Data (after conversion)========")
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)
	#print(df.head())

	#X is the set of features
	X = df[df.columns[:-1]].values
	if "Random" in filename:
		X = np.delete(X, 0, 1)
	#y is the set of classes
	y = df['class'].values
	
	#Use smote to fix the undersampling, this generates new data for positive and neutral based off of the previous data
	X_resampled, y_resampled = SMOTE().fit_resample(X, y)

	print("========After Smote Padding========")
	print(sorted(collections.Counter(y_resampled).items()))

	return X_resampled,y_resampled

#This undersamples the data to balance classes and writes it
def undersample(filename):
	print("Undersample on on "+filename)
	#Load in the csv of the data from NNData
	df = pd.read_csv("NN_SVMData/"+filename)
	print("========Head of the Data========")
	print(df.head())
	#this shows there is a harsh imbalance of the classes (positive, negative, neutral)
	print("========Count of Classes========")
	print(df['class'].value_counts())

	#This is to convert the classes to numbers
	#print("========Head of the Data (after conversion)========")
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)
	#print(df.head())

	#X is the set of features
	X = df[df.columns[:-1]].values
	#y is the set of classes
	y = df['class'].values

	rus = RandomUnderSampler(random_state=0)
	#rus.fit(X, y)
	X_resampled, y_resampled = rus.fit_resample(X, y)
	print("========After Undersampling Padding========")
	print(sorted(collections.Counter(y_resampled).items()))
	
	if "Random" in filename:
		X_resampled = np.delete(X_resampled, 0, 1)

	return X_resampled,y_resampled

def video_seperate_gender(filename):
	df = pd.read_csv("NN_SVMData/"+filename)
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)

	ran = pd.read_csv("Random_Video_Analytics.csv")
	ran = ran.values

	seq = pd.read_csv("Seq_Video_Analytics.csv")
	seq = seq.values
	#print(df.head())

	#X is the set of features
	X = df[df.columns[:-1]].values
	#y is the set of classes
	y = df['class'].values

	male_y = []
	male_X = []
	female_y = []
	female_X = []

	if 'Random' not in  filename:
		for j in range(len(X)//18):
			for i in range(18):
				if seq[(j//11)][0] == "Female":
					female_X.append(X[(j*18)+i])
					female_y.append(y[(j*18)+i])
				else:
					male_X.append(X[(j*18)+i])
					male_y.append(y[(j*18)+i])
	else:
		X = np.delete(X, 0, 1)
		for j in range(len(X)//18):
			for i in range(18):
				if ran[j][0] == "Female":
					female_X.append(X[(j*18)+i])
					female_y.append(y[(j*18)+i])
				else:
					male_X.append(X[(j*18)+i])
					male_y.append(y[(j*18)+i])
	return np.asarray(male_X), np.asarray(male_y), np.asarray(female_X), np.asarray(female_y)

def tester_seperate_gender(filename):
	df = pd.read_csv("NN_SVMData/"+filename)
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)

	an = pd.read_csv("TesterInformation.csv")
	an = an.values

	#X is the set of features
	X = df[df.columns[:-1]].values
	#y is the set of classes
	y = df['class'].values

	male_y = []
	male_X = []
	female_y = []
	female_X = []

	if "Random" in filename:
		X = np.delete(X, 0, 1)
	
	for j in range(len(X)//18):
		for i in range(18):
			if an[i][2] == "Female":
				female_X.append(X[(j*18)+(i-1)])
				female_y.append(y[(j*18)+(i-1)])
			else:
				male_X.append(X[(j*18)+(i-1)])
				male_y.append(y[(j*18)+(i-1)])


	return np.asarray(male_X), np.asarray(male_y), np.asarray(female_X), np.asarray(female_y)

def video_seperate_race(filename):
	df = pd.read_csv("NN_SVMData/"+filename)
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)

	ran = pd.read_csv("Random_Video_Analytics.csv")
	ran = ran.values

	seq = pd.read_csv("Seq_Video_Analytics.csv")
	seq = seq.values
	#print(df.head())

	#X is the set of features
	X = df[df.columns[:-1]].values
	#y is the set of classes
	y = df['class'].values

	race_X = defaultdict(list)
	race_y = defaultdict(list)


	if 'Random' not in  filename:
		for j in range(len(X)//18):
			for i in range(18):
				race_X[seq[(j//11)][1]].append(X[(j*18)+i])
				race_y[seq[(j//11)][1]].append(y[(j*18)+i])
	else:
		X = np.delete(X, 0, 1)
		for j in range(len(X)//18):
			for i in range(18):
				race_X[ran[j][1]].append(X[(j*18)+i])
				race_y[ran[j][1]].append(y[(j*18)+i])
				
	return race_X, race_y

def tester_seperate_race(filename):
	df = pd.read_csv("NN_SVMData/"+filename)
	d = {"class":{"Positive":0,"Negative":1,"Neutral":2}}
	df.replace(d,inplace=True)

	an = pd.read_csv("TesterInformation.csv")
	an = an.values

	#X is the set of features
	X = df[df.columns[:-1]].values
	#y is the set of classes
	y = df['class'].values

	race_X = defaultdict(list)
	race_y = defaultdict(list)

	if "Random" in filename:
		X = np.delete(X, 0, 1)

	for j in range(len(X)//18):
		for i in range(18):
			race_X[an[i][3]].append(X[(j*18)+(i-1)])
			race_y[an[i][3]].append(y[(j*18)+(i-1)])

	return race_X, race_y

#This writes the data back, now that it has been reformated, in arff format
def writeData(X,y,type):
	file = open("WekaData/Weka"+type+"AllSimple.arff","w")
	file1 = open("PostSmoteData/EmotionData"+type+"All.csv","w")
	attributes = ["AU1","AU2","AU4","AU6","AU7","AU9","AU12","AU15","AU16","AU20","AU23","AU26","Left","Lower","Right","Upper","class"]
	file.write("@relation emotion\n")
	file.write("\n")

	#Write the attributes in the correct format
	for i in range(len(attributes)-1):
		file.write("@attribute "+attributes[i]+" numeric\n")
		file1.write(attributes[i]+",")


	file.write("@attribute "+attributes[len(attributes)-1]+"{Positive,Negative,Neutral}\n")
	file1.write(attributes[len(attributes)-1]+"\n")

	file.write("\n@data\n")
	
	for i in range(len(y)):
		temp = X[i]
		st = ','.join(str(e) for e in temp)
		if y[i] == 0:
			st += ",Positive"
		elif y[i] == 1:
			st += ",Negative"
		else:
			st += ",Neutral"
		file1.write(st+"\n")
		file.write(st+"\n")
	file1.close()
	file.close()

def write_csv(X,y,type):
	file = open("UnderSampledData/EmotionData"+type+".csv","w")
	attributes = ["AU1","AU2","AU4","AU6","AU7","AU9","AU12","AU15","AU16","AU20","AU23","AU26","Left","Lower","Right","Upper","class"]

	for i in range(len(attributes)-1):
		file.write(attributes[i]+",")	

	file.write(attributes[len(attributes)-1]+"\n")

	for i in range(len(y)):
		temp = X[i]
		st = ','.join(str(e) for e in temp)
		if y[i] == 0:
			st += ",Positive"
		elif y[i] == 1:
			st += ",Negative"
		else:
			st += ",Neutral"
		file.write(st+"\n")

	file.close()

'''
#This function tests the data with a convolutional neural network
def makeNN(X,y):
	print("============Testing data With Deep Learning============")
	print(tf.__version__)

	print("Shape of raw data:",X.shape,y.shape)
	#This partitions the indices randomly for 80% test data and 20% training data
	indices = np.random.permutation(y.shape[0])
	index = int(len(X)*0.8)
	training_idx, test_idx = indices[:index], indices[index:]
	#print(indices)

	#create training data
	train_data, train_labels = X[training_idx], y[training_idx]
	test_data,test_labels = X[test_idx], y[test_idx]

	print("Shape of test data before:",test_data.shape,test_labels.shape)
	print("Shape of train data before:",train_data.shape,train_labels.shape)

	train_data = train_data.reshape(len(train_data),16,1,1)
	test_data = test_data.reshape(len(test_data),16,1,1)
	#Create testing data by


	#print the shape of the data
	print("Shape of test data:",test_data.shape,test_labels.shape)
	print("Shape of train data:",train_data.shape,train_labels.shape)

	

	#Create network, input is already vectors, so no need to flatten, using relu activation function and softmax classifier
	
	model = keras.models.Sequential([
	    keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu, input_shape=(16,1,1)),
	    keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(2, 2), activation=tf.nn.relu),
		keras.layers.Flatten(),
		keras.layers.Dense(1000, activation='relu'),
		keras.layers.Dense(3, activation='softmax')
	])

	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	
	network = model.fit(train_data,
          train_labels, 
          epochs=100, 
          validation_data=(test_data, test_labels),
          verbose=2)
	
	#plot the validation accuracy over epochs
	plt.plot(network.history['acc'])
	plt.plot(network.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	#plt.savefig("Figures/RanFemaleTester.png")
'''
def makeSVM(X,y, smote):
	if smote:
		X, y = SMOTE().fit_resample(X, y)

	scaler = StandardScaler()
	scaler.fit(X)

	X = scaler.transform(X)

	C_2d_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	gamma_2d_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	classifiers = []
	best = [0,0,0]
	y_true, y_pred = y, None

	for C in tqdm(C_2d_range, desc="C range"):
		for gamma in tqdm(gamma_2d_range, desc="gamma range", leave=False):
			clf = SVC(C=C, gamma=gamma)
			clf.fit(X, y)
			classifiers.append((C, gamma, clf))
			#print("Model for gamma {} and C {}".format(gamma, C))
			predicted = cross_val_predict(clf, X, y, cv=5)
			#c = cross_val_score(clf, X, y, scoring='accuracy',cv=5)
			a = metrics.accuracy_score(y, predicted)
			if a > best[0]:
				best = [a,gamma, C]
				y_true, y_pred = y, predicted
	print("Best was {} with gamma {} and C {}".format(best[0], best[1], best[2]))
	print(metrics.classification_report(y_true, y_pred, target_names=['Positive', 'Negative', 'Neutral']))
	print(y[0], len(y_pred))
	plotModelMatrix(y, y_pred, "RandomSVM", "", "Random", "Random Video SVM Confusion Matrix")
	
def plotModelMatrix(y_true,y_pred, name, tester,type, title):
    #print(y_true, y_pred)
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    #print('{0}:'.format(i))
    print(cm)
    print(len(y_true))
    #Print the accuracies of all emotions    
    
    acc=accuracy_score(y_true, y_pred)
    acc1 = cm[0,0]/sum(cm[0,])
    acc2 = cm[1,1]/sum(cm[1,])
    acc3 = cm[2,2]/sum(cm[2,])
    acc=accuracy_score(y_true, y_pred)
    print('accuracy of POS:{0}'.format(acc1))
    print('accuracy of NEG:{0}'.format(acc2))
    print('accuracy of NEU:{0}'.format(acc3))
    print('accuracy:{0}'.format(acc))
    
    acc = accuracy_score(y_true, y_pred)
    print('accuracy:{0}'.format(acc))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % (p)
    cm = pd.DataFrame(cm, index=['Positive','Negative','Neutral'], columns=['Positive','Negative','Neutral']) #here you can change the label 
    cm.index.name = 'Label'
    cm.columns.name = 'Prediction'
    figsize=(3,3)
    fig, ax = plt.subplots()
    fig.tight_layout()
    print(ax)
    print(fig)
    sns.heatmap(cm, cmap=plt.cm.Blues ,square=True ,annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
    plt.title(title)  #If you need to be able to change the name of the martix here
    plt.savefig(r'ConfusionMatrices'+type+'/'+name+str(tester)+'ResponseConfusionMatrix.png')  #Here you can choose where to save and only save the image of the matrix
    plt.show()


if __name__ == "__main__":

	X_seq,y_seq = doSMOTE("EmotionDataSequentialAll.csv") #undersample("EmotionDataSequentialAll.csv")
	X_ran,y_ran = doSMOTE("EmotionDataRandomAll.csv") #undersample("EmotionDataRandomAll.csv")

	X_male_v, y_male_v, X_female_v, y_female_v = video_seperate_gender("EmotionDataRandomAll.csv")
	X_male_u, y_male_u, X_female_u, y_female_u = tester_seperate_gender("EmotionDataRandomAll.csv")

	X_race_v, y_race_v = video_seperate_race("EmotionDataRandomAll.csv")
	X_race_u, y_race_u = tester_seperate_race("EmotionDataRandomAll.csv")
	'''
	for (X_k, X_v),(y_k, y_v) in zip(X_race_v.items(), y_race_v.items()):
		print("model for {}".format(X_k))
		makeSVM(X_v, y_v, True)
	'''
	makeSVM(X_ran, y_ran, True)
	
	'''
	
	#write all data to csv files
	write_csv(X_male_v, y_male_v, "RandomVideoMale")
	write_csv(X_female_v, y_female_v, "RandomVideoFemale")

	write_csv(X_male_u, y_male_u, "RandomTesterMale")
	write_csv(X_female_u, y_female_u, "RandomTesterFemale")

	for key,v in X_race_v.items():
		write_csv(X_race_v[key], y_race_v[key], "RandomVideo"+key)
	
	for key,v in X_race_u.items():
		write_csv(X_race_u[key], y_race_u[key], "RandomTester"+key)
	
	'''
	exit()
	
	#do smote for sequential, random and all
	X_seq,y_seq = doSMOTE("EmotionDataSequentialAll.csv")
	X_ran,y_ran = doSMOTE("EmotionDataRandomAll.csv")
	X_all,y_all = doSMOTE("EmotionDataAll.csv")
	X_ran_per, y_ran_per = doSMOTE("EmotionDataRandomPercievedAll.csv")
	X_seq_per, y_seq_per = doSMOTE("EmotionDataSequentialPercievedAll.csv")

	data = [[X_seq,y_seq],[X_ran,y_ran],[X_all,y_all],[X_seq_per, y_seq_per],[X_ran_per, y_ran_per]]

	#Now write the data for Weka

	writeData(X_seq,y_seq,"Sequential")
	writeData(X_ran,y_ran,"Random")
	writeData(X_ran_per,y_ran_per, "RandomPercieved")
	writeData(X_seq_per,y_seq_per, "SequentialPercieved")
	
	#From here, want to test a NN, and want to write this data, the NN is just for fun to see how it works on thie data

	#makeNN(X_ran,y_ran)

	#LinearSVM
	
	results = {"Seq":[],"Ran":[],"All":[],"Seq_Per":[],"Ran_Per":[]}
	iterlist = [20000,30000,30000,30000,20000]
	keys = list(results.keys())
	for i in range(len(data)):
		print("Model",keys[i])
		results[keys[i]].append(makeSVM(data[i][0],data[i][1],iterlist[i]))
	print(results)