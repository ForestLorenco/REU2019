import pandas as pd
import numpy as np

PATH = "PostSmoteData/"

WRITEPATH = "NormalizedData/"

WEKAPATH = "WekaNormData/"

def write_data(values, cols, dataset):
    """
    Write data and remove the data data points with all 0s
    :param values:
    :param cols:
    :param dataset:
    :return:
    """
    file = open(WRITEPATH+dataset, "w")
    weka = open(WEKAPATH+dataset[0:len(dataset)-3]+"arff", "w")

    weka.write("@relation emotion\n")
    weka.write("\n")


    for i in range(len(cols)-1):
        weka.write("@attribute " + cols[i] + " numeric\n")
        file.write(cols[i]+",")

    weka.write("@attribute " + cols[-1] + "{Positive,Negative,Neutral}\n")
    file.write(cols[-1]+"\n")

    weka.write("\n@data\n")

    for v in values:
        l = np.sum(v[0:len(v)-1])
        if l != 0:
            for i in range(len(v)-1):
                weka.write(str(v[i]) + ",")
                file.write(str(v[i]) + ",")
            weka.write(str(v[-1]) + "\n")
            file.write(str(v[-1]) + "\n")
    file.close()
    weka.close()

def normalize(data, name):
    """
    Normalizes the data by making all videos length of 1
    :param data:
    :return:
    """
    cols = list(data.columns)
    vals = data.values
    for i in range(len(vals)):
        v = vals[i]
        l = np.sum(v[0:len(v)-1])
        if l != 0:
            t = v[0:len(v)-1]/l
            v = np.append(t, v[-1])
            vals[i] = v
    write_data(vals, cols, name)




if __name__ == "__main__":
    datasets = ["EmotionDataRandomAll.csv", "EmotionDataRandomPercievedAll.csv", "EmotionDataSequentialAll.csv", "EmotionDataSequentialPercievedAll.csv"]
    for d in datasets:
        data = pd.read_csv(PATH+d)
        normalize(data, d)