import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns

def makeModelData(correctfile, predfile):
        y_true = []
        y_pred = []
        cfile = open("ResponseDataModels/"+correctfile,"r")
        rfile = open("ResponseDataModels/"+predfile,"r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)
        return (y_true,y_pred)


def plotModelMatrix(y_true,y_pred, name, tester,type, title):
    #print(y_true, y_pred)
    labels = ["1", "2", "3"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    #print('{0}:'.format(i))
    print(cm)
    print(len(y_true))
    #Print the accuracies of all emotions    
    '''  
    acc=accuracy_score(y_true, y_pred)
    acc1 = cm[0,0]/sum(cm[0,])
    acc2 = cm[1,1]/sum(cm[1,])
    acc3 = cm[2,2]/sum(cm[2,])
    acc4 = cm[3,3]/sum(cm[3,])
    acc5 = cm[4,4]/sum(cm[4,])
    acc6 = cm[5,5]/sum(cm[5,])
    acc=accuracy_score(y_true, y_pred)
    print('accuracy of ANG:{0}'.format(acc1))
    print('accuracy of DIS:{0}'.format(acc2))
    print('accuracy of FEA:{0}'.format(acc3))
    print('accuracy of HAP:{0}'.format(acc4))
    print('accuracy of NEU:{0}'.format(acc5))
    print('accuracy of SAD:{0}'.format(acc6))
    print('accuracy:{0}'.format(acc))
    '''

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
    #plt.show()

def plotMatrix(y_true,y_pred, name, tester,type, title):
    #print(y_true, y_pred)
    labels = ["1", "2", "3" , "4" , "5" , "6"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    #print('{0}:'.format(i))
    print(cm)
    print(len(y_true))
    #Print the accuracies of all emotions      
    acc=accuracy_score(y_true, y_pred)
    acc1 = cm[0,0]/sum(cm[0,])
    acc2 = cm[1,1]/sum(cm[1,])
    acc3 = cm[2,2]/sum(cm[2,])
    acc4 = cm[3,3]/sum(cm[3,])
    acc5 = cm[4,4]/sum(cm[4,])
    acc6 = cm[5,5]/sum(cm[5,])
    acc=accuracy_score(y_true, y_pred)
    print('accuracy of ANG:{0}'.format(acc1))
    print('accuracy of DIS:{0}'.format(acc2))
    print('accuracy of FEA:{0}'.format(acc3))
    print('accuracy of HAP:{0}'.format(acc4))
    print('accuracy of NEU:{0}'.format(acc5))
    print('accuracy of SAD:{0}'.format(acc6))
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
    cm = pd.DataFrame(cm, index=['ANG','DIS','FEA','HAP','NEU','SAD'], columns=['ANG','DIS','FEA','HAP','NEU','SAD']) #here you can change the label 
    cm.index.name = 'Label'
    cm.columns.name = 'Prediction'
    figsize=(6,6)
    fig, ax = plt.subplots()
    fig.tight_layout()
    print(ax)
    print(fig)
    sns.heatmap(cm, cmap=plt.cm.Blues ,square=True ,annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
    plt.title(title)  #If you need to be able to change the name of the martix here
    plt.savefig(r'ConfusionMatrices'+type+'/'+name+str(tester)+'ResponseConfusionMatrix.png')  #Here you can choose where to save and only save the image of the matrix
    #plt.show()

    

def main():
    y_true = []
    y_pred = [] 
    '''
    #Sequential
    #Plot the confusion matrices of each tester individually
    for i in range(1,7):
        cfile = open("ResponseDataSequential/CorrectResponse.txt","r")
        rfile = open("ResponseDataSequential/ResponseTester"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

        plotMatrix(y_true,y_pred,"Tester",i,"Sequential")
        y_pred.clear()
        y_true.clear()


    for i in range(1,13):
        cfile = open("ResponseDataSequential/CorrectResponse.txt","r")
        rfile = open("ResponseDataSequential/ResponseParticipant"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

        plotMatrix(y_true,y_pred,"Participant", i,"Sequential")
        y_pred.clear()
        y_true.clear()
    '''

    #Plot the confusion matrices of all testers in one
    for i in range(1,7):
        cfile = open("ResponseDataSequential/CorrectResponse.txt","r")
        rfile = open("ResponseDataSequential/ResponseTester"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

    for i in range(1,13):
        cfile = open("ResponseDataSequential/CorrectResponse.txt","r")
        rfile = open("ResponseDataSequential/ResponseParticipant"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

    plotMatrix(y_true,y_pred,"","All","Sequential", "Participant Response Random Stimilus")

    #Sequential
    '''
    #Plot the confusion matrices of each tester individually
    for i in range(1,7):
        cfile = open("ResponseDataRandom/CorrectResponse.txt","r")
        rfile = open("ResponseDataRandom/ResponseTester"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

        plotMatrix(y_true,y_pred,"Tester",i,"Random")
        y_pred.clear()
        y_true.clear()

    for i in range(1,13):
        cfile = open("ResponseDataRandom/CorrectResponse.txt","r")
        rfile = open("ResponseDataRandom/ResponseParticipant"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

        plotMatrix(y_true,y_pred,"Participant",i,"Random")
        y_pred.clear()
        y_true.clear()
    '''

    #Plot the confusion matrices of all testers in one
    for i in range(1,7):
        cfile = open("ResponseDataRandom/CorrectResponse.txt","r")
        rfile = open("ResponseDataRandom/ResponseTester"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

    for i in range(1,13):
        cfile = open("ResponseDataRandom/CorrectResponse.txt","r")
        rfile = open("ResponseDataRandom/ResponseParticipant"+str(i)+".txt","r")
        clines = cfile.readlines()
        rlines = rfile.readlines()
        for line in clines:
            s = line[:1]                                   
            y_true.append(s)
        for line in rlines:
            s = line[:1]
            y_pred.append(s)

    plotMatrix(y_true,y_pred,"","All","Random","Participant Response Sequential Stimilus")

    #Make the model matrices
    temp = makeModelData("RandomCorrect.csv","RandomPred.csv")
    plotModelMatrix(temp[0],temp[1],"","ModelRan","Random","Model Classification Random Stimilus")

    temp = makeModelData("RandomResponseCorrect.csv","RandomResponsePred.csv")
    plotModelMatrix(temp[0],temp[1],"","ModelRanRes","Random","Model Classification Random Stimilus (Correct Response Only)")

    temp = makeModelData("SequentialResponseCorrect.csv","SequentialResponsePred.csv")
    plotModelMatrix(temp[0],temp[1],"","ModelSeqRes","Sequential","Model Classification Sequential Stimilus (Correct Response Only)")


    #Make the model matrices
    temp = makeModelData("SequentialCorrect.csv","SequentialPred.csv")
    plotModelMatrix(temp[0],temp[1],"","ModelSeq","Sequential","Model Classification Sequential Stimilus")
    print('finished')


    
main()