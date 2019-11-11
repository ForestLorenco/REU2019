import pandas as pd
import collections
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np

WEKAPATH = "LOO_data/"


# This writes the data back, now that it has been reformated, in arff format
def write_data(X, y, type, tester, train):
    file = open("LOO_data/" + type + tester + train+".arff", "w")
    attributes = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU12", "AU15", "AU16", "AU20", "AU23", "AU26", "Left",
                  "Lower", "Right", "Upper", "class"]
    file.write("@relation emotion\n")
    file.write("\n")

    # Write the attributes in the correct format
    for i in range(len(attributes) - 1):
        file.write("@attribute " + attributes[i] + " numeric\n")

    file.write("@attribute " + attributes[len(attributes) - 1] + "{Positive,Negative,Neutral}\n")

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
        file.write(st + "\n")
    file.close()

def doSMOTE(filename, type):
    print("SMOTE on " + filename)
    # Load in the csv of the data from NNData
    df = pd.read_csv("LOO_data/" + filename)
    for i in range(1,19):
        test = df[df["Tester"] == "Tester"+str(i)]
        temp = df[df["Tester"] != "Tester"+str(i)]
        print("Leaving out Tester"+str(i))
        print("========Head of the Data========")
        print(df.head())
        # this shows there is a harsh imbalance of the classes (positive, negative, neutral)
        print("========Count of Classes========")
        print(df['class'].value_counts())
        temp = temp.drop("Tester", axis=1)

        # This is to convert the classes to numbers
        # print("========Head of the Data (after conversion)========")
        d = {"class": {"Positive": 0, "Negative": 1, "Neutral": 2}}
        temp.replace(d, inplace=True)
        # print(df.head())

        # X is the set of features
        X_train = temp[temp.columns[:-1]].values
        # y is the set of classes
        y_train = temp['class'].values

        # X is the set of features
        X_test = test[test.columns[:-1]].values
        # y is the set of classes
        y_test = test['class'].values

        # Use smote to fix the undersampling, this generates new data for positive and neutral based off of the previous data
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

        print("========After Smote Padding========")
        print(sorted(collections.Counter(y_resampled).items()))

        write_data(X_resampled, y_resampled, type, "Tester"+str(i),"train")
        write_data(X_test, y_test, type,"Tester"+str(i), "test")




if __name__ == "__main__":
    doSMOTE("EmotionDataRandomAll.csv", "Random")
    doSMOTE("EmotionDataRandomPercievedAll.csv", "Random")
    doSMOTE("EmotionDataSequentialAll.csv", "Sequential")
    doSMOTE("EmotionDataSequentialPercievedAll.csv", "Sequential")
