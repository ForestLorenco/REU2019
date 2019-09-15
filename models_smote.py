from __future__ import absolute_import, division, print_function, unicode_literals

#this is for oversampling, using smote
import pandas as pd
import collections
from imblearn.over_sampling import SMOTE, ADASYN

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

#This function tests the data with a 2 layer neural network
def makeNN(X,y):
	print("============Testing data With Deep Learning============")
	print(tf.__version__)

	print("Shape of raw data:",X.shape,y.shape)
	#This partitions the indices randomly for 80% test data and 20% training data
	indices = np.random.permutation(y.shape[0])
	training_idx, test_idx = indices[:844], indices[844:]
	#print(indices)

	#create training data
	train_data, train_labels = X[training_idx], y[training_idx]
	test_data,test_labels = X[test_idx], y[test_idx]
	#Create testing data by


	#print the shape of the data
	print("Shape of test data:",test_data.shape,test_labels.shape)
	print("Shape of train data:",train_data.shape,train_labels.shape)




	#Create network, input is already vectors, so no need to flatten, using relu activation function and softmax classifier
	
	model = keras.models.Sequential([
	    keras.layers.Dense(32,activation=tf.nn.relu),
	    keras.layers.BatchNormalization(),
	    keras.layers.Dropout(0.2), #Helps overfitting
	    keras.layers.Dense(64,activation=tf.nn.relu),
	    keras.layers.BatchNormalization(),
	    keras.layers.Dropout(0.2),
	    keras.layers.Dense(3,activation=tf.nn.softmax)
	])

	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	
	network = model.fit(train_data,
          train_labels, 
          epochs=25, 
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


def main():
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
	
	
main()