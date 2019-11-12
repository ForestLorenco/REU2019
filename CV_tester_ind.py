import pandas as pd
import numpy as np
from collections import defaultdict

# This writes the data back, now that it has been reformated, in arff format
def write_data(X, attr, type, fold, train):
    file = open("CV_weka_data/" + type + fold + train+".arff", "w")
    attributes = attr
    file.write("@relation emotion\n")
    file.write("\n")

    # Write the attributes in the correct format
    for i in range(len(attributes) - 1):
        file.write("@attribute " + attributes[i] + " numeric\n")

    file.write("@attribute " + attributes[len(attributes) - 1] + "{Positive,Negative,Neutral}\n")

    file.write("\n@data\n")

    for i in X:
        st = ','.join(str(e) for e in i[1:])
        file.write(st + "\n")
    file.close()

def get_tester_results(test, i, type):
    global tp
    df = pd.read_csv("CV_classifications/"+type+i)
    vals = df.values
    for val, t in zip(vals,test):
        if val[3] == "+":
            tp[t[0]][0] += 1
        tp[t[0]][1] += 1


def CV(filename, type):
    np.random.seed(12345)
    df = pd.read_csv("LOO_data/" + filename)

    df = df.sample(frac=1)
    #print(df)
    data = df.values
    attr = df.columns
    for i in range(10):
        l = len(data)
        #print("==============Fold {}===========".format(i))
        train, test = np.concatenate((data[0:int((i*0.1)*l)], data[int(((i+1)*0.1)*l):]), axis=0),data[int((i*0.1)*l):int(((i+1)*0.1)*l)]
        #print(len(train), len(test), len(data))

        #write_data(train, attr[1:], str(i+1), type, "train")
        #write_data(test, attr[1:], str(i+1), type, "test")
        get_tester_results(test, str(i+1), type)
        break


tp = {}

if __name__ == "__main__":
    for i in range(1,19):
        tp["Tester"+str(i)] = [0,0]
    CV("EmotionDataRandomAll.csv", "Ran")
    print("Random")
    for key in tp.keys():
        print("{} rate is {}".format(key, 1-(tp[key][0]/tp[key][1])))

    for i in range(1,18):
        tp["Tester"+str(i)] = [0,0]
    CV("EmotionDataSequentialAll.csv", "Seq")
    print("Seq")
    for key in tp.keys():
        print("{} rate is {}".format(key, 1- (tp[key][0] / tp[key][1])))

    '''
    for i in range(1, 11):
        file = open("CV_classifications/Seq"+str(i), "w")
        file.close()
    '''