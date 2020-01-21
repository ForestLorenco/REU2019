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

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

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
		for j in range(len(X)//18):
			for i in range(18):
				if ran[j][0] == "Female":
					female_X.append(X[(j*18)+i])
					female_y.append(y[(j*18)+i])
				else:
					male_X.append(X[(j*18)+i])
					male_y.append(y[(j*18)+i])
	return male_X, male_y, female_X, female_y

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

	
	for j in range(len(X)//18):
		for i in range(18):
			if an[i][2] == "Female":
				female_X.append(X[(j*18)+(i-1)])
				female_y.append(y[(j*18)+(i-1)])
			else:
				male_X.append(X[(j*18)+(i-1)])
				male_y.append(y[(j*18)+(i-1)])


	return male_X, male_y, female_X, female_y

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
				race_y[seq[(j//11)][1]].append(X[(j*18)+i])
	else:
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


#This function takes the histories of a network, and plkots the validation accuracy over epochs
def plot_history(histories, key='acc'):
	plt.figure(figsize=(16,10))

	for name, history in histories:
		val = plt.plot(history.epoch, history.history['val_'+key],
				'--', label=name.title()+' Val')
		plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
				label=name.title()+' Train')

	plt.xlabel('Epochs')
	plt.ylabel(key.replace('_',' ').title())
	plt.legend()
	plt.xlim([0,max(history.epoch)])
	input()

#This function tests the data with a convolutional neural network
def makeNN(X,y):
	print("============Testing data With Deep Learning============")
	print(tf.__version__)

	print("Shape of raw data:",X.shape,y.shape)
	#This partitions the indices randomly for 80% test data and 20% training data
	indices = np.random.permutation(y.shape[0])
	training_idx, test_idx = indices[:692], indices[692:]
	#print(indices)

	#create training data
	train_data, train_labels = X[training_idx], y[training_idx]
	test_data,test_labels = X[test_idx], y[test_idx]

	train_data = np.delete(train_data, 0, 1)
	test_data = np.delete(test_data, 0, 1)

	print("Shape of test data before:",test_data.shape,test_labels.shape)
	print("Shape of train data before:",train_data.shape,train_labels.shape)

	train_data = train_data.reshape(692,16,1,1)
	test_data = test_data.reshape(172,16,1,1)
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
	#print(network.history.keys())
	#plot_history([('Network', network)])

def makeSVM(X,y,max_iter):

	#Normalize the data
	scaler = StandardScaler()
	scaler.fit(X)

	n_X = scaler.transform(X)
	#Converges at 20000
	clf = svm.LinearSVC(max_iter=max_iter)
	print("Creating LinearSVC model...")
	clf.fit(n_X, y)

	print("Evaluating LinearSVC model with 5 folds...")
	return cross_val_score(clf, n_X, y, scoring='accuracy',cv=5)


if __name__ == "__main__":

	X_seq,y_seq = undersample("EmotionDataSequentialAll.csv")
	X_ran,y_ran = undersample("EmotionDataRandomAll.csv")

	X_male_v, y_male_v, X_female_v, y_female_v = video_seperate_gender("EmotionDataRandomAll.csv")
	X_male_u, y_male_u, X_female_u, y_female_u = tester_seperate_gender("EmotionDataRandomAll.csv")

	X_race_v, y_race_v = video_seperate_race("EmotionDataRandomAll.csv")
	X_race_u, y_race_u = tester_seperate_race("EmotionDataRandomAll.csv")


	print(y_race_v)

	#makeNN(X_ran,y_ran)

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