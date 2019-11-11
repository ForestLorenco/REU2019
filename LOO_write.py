# This file is for processing the data for leave one out
def LOODataSet(num_testers, mispercievedSeq, mispercievedRan):
    AOIfile = open("AOIMetricsRandom.csv", "r")
    # AOIfileR=open("AOIMetricsRandom.csv","r")
    file = open("LOO_data/EmotionDataRandomAll.csv", "w")
    fileP = open("LOO_data/EmotionDataRandomPercievedAll.csv", "w")
    fileA = open("LOO_data/EmotionDataAll.csv", "w")
    attributes = ["Tester","AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU12", "AU15", "AU16", "AU20", "AU23", "AU26", "Left",
                  "Lower", "Right", "Upper", "class"]

    mispercieved = mispercievedRan
    keys = list(mispercieved.keys())

    for element in attributes:
        if element != "class":
            fileA.write(element + ",")

        else:
            fileA.write(element)
    fileA.write("\n")

    for element in attributes:
        if element != "class":
            file.write(element + ",")
            fileP.write(element + ",")

        else:
            file.write(element)
            fileP.write(element)
    file.write("\n")
    fileP.write("\n")

    for k in range(2):

        index = 0
        done = False
        while not done:

            line = AOIfile.readline()

            if line == "":
                break
            if "_DIS_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester"+str(i+1)+","+st
                    file.write(st + ",Negative\n")

                    fileA.write(st + ",Negative\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Negative\n")

            if "_HAP_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester"+str(i+1)+","+st
                    file.write(st + ",Positive\n")

                    fileA.write(st + ",Positive\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Positive\n")

            if "_ANG_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester"+str(i+1)+","+st
                    file.write(st + ",Negative\n")

                    fileA.write(st + ",Negative\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Negative\n")

            if "_FEA_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester"+str(i+1)+","+st
                    file.write(st + ",Negative\n")

                    fileA.write(st + ",Negative\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Negative\n")

            if "_NEU_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester"+str(i+1)+","+st
                    file.write(st + ",Neutral\n")

                    fileA.write(st + ",Neutral\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Neutral\n")

            if "_SAD_" in line:
                index += 1
                line = AOIfile.readline()
                for i in range(num_testers):
                    temp = [0] * 16
                    line = AOIfile.readline()
                    part = line.split(",")
                    for j in range(16):
                        if part[j + 2] != "" and part[j + 2] != "\n":
                            temp[j] += float(part[j + 2])

                    st = ','.join(str(e) for e in temp)
                    st = "Tester" + str(i + 1) + "," + st
                    file.write(st + ",Negative\n")

                    fileA.write(st + ",Negative\n")
                    if index not in mispercieved[keys[i]]:
                        fileP.write(st + ",Negative\n")
        if k == 0:
            file.close()
            file = open("NN_SVMData/EmotionDataRandomAll.csv", "w")
            fileP.close()
            fileP = open("NN_SVMData/EmotionDataRandomPercievedAll.csv", "w")
            AOIfile.close()
            AOIfile = open("AOIMetricsRandom.csv", "r")
            for element in attributes:
                if element != "class":
                    file.write(element + ",")
                    fileP.write(element + ",")
                else:
                    file.write(element)
                    fileP.write(element)
            mispercieved = mispercievedRan
            keys = list(mispercieved.keys())
            file.write("\n")
            fileP.write("\n")

    fileA.close()
    file.close()
    fileP.close()
    AOIfile.close()

if __name__ == "__main__":
    mispercievedSeq = {
        'Participant1': [1, 2, 3, 6, 8, 10, 12, 15, 16, 22, 24, 26, 30, 37, 39, 45, 46, 48, 49, 52, 55, 63, 70, 72, 74,
                         75, 76, 78, 80, 83, 87, 88, 92],
        'Participant2': [5, 9, 10, 15, 20, 22, 26, 31, 32, 33, 39, 42, 45, 48, 49, 51, 52, 59, 63, 67, 69, 71, 72, 74,
                         75, 78, 79, 80, 81, 92],
        'Participant3': [1, 6, 10, 15, 16, 17, 20, 21, 23, 24, 27, 29, 33, 35, 39, 41, 42, 45, 46, 49, 52, 54, 55, 57,
                         58, 60, 64, 65, 68, 69, 70, 71, 74, 78, 83, 86, 88, 92],
        'Participant4': [9, 10, 22, 24, 26, 27, 30, 42, 48, 52, 54],
        'Participant5': [4, 20, 22, 23, 46, 52, 53, 55, 66, 72],
        'Participant6': [5, 9, 12, 15, 24, 27, 30, 33, 39, 48, 51, 52, 54, 55, 58, 60, 63, 72, 74, 75, 78, 81, 83, 88,
                         92],
        'Participant7': [2, 3, 5, 6, 9, 12, 15, 16, 20, 21, 22, 24, 26, 27, 30, 32, 42, 45, 46, 52, 57, 62, 70, 72, 74,
                         75, 78, 81, 83, 91, 92],
        'Participant8': [2, 4, 5, 20, 23, 26, 27, 33, 36, 40, 42, 45, 46, 48, 49, 51, 52, 55, 70, 72, 78, 92],
        'Participant9': [2, 6, 9, 12, 15, 17, 22, 24, 27, 28, 29, 30, 33, 39, 40, 42, 45, 46, 48, 49, 51, 56, 57, 66,
                         69, 70, 72, 74, 78, 81, 83, 84, 88, 92],
        'Participant10': [1, 6, 7, 10, 12, 15, 20, 22, 29, 30, 32, 33, 41, 42, 45, 48, 51, 63, 68, 69, 72, 78, 80, 83,
                          86, 88, 89, 91, 92],
        'Tester1': [1, 2, 6, 9, 12, 15, 17, 24, 26, 27, 30, 33, 40, 42, 45, 46, 47, 48, 51, 52, 58, 59, 72, 74, 75, 78],
        'Tester2': [10, 20, 22, 24, 26, 27, 30, 32, 33, 40, 42, 44, 45, 46, 48, 49, 52, 56, 57, 58, 70, 72, 74, 75, 78,
                    81, 91, 92],
        'Tester3': [5, 6, 8, 15, 20, 24, 26, 30, 40, 42, 45, 46, 48, 49, 52, 70, 72, 76, 78, 80, 89],
        'Tester4': [4, 5, 6, 12, 17, 21, 24, 26, 27, 29, 30, 33, 39, 40, 42, 45, 46, 48, 51, 52, 58, 59, 62, 70, 72, 78,
                    83, 86, 92], 'Tester5': [2, 6, 20, 22, 23, 24, 29, 41, 46, 47, 48, 54, 55, 58, 66, 70, 77, 78, 80],
        'Tester6': [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 23, 24, 27, 29, 30, 32, 33, 36, 39, 40, 42,
                    45, 46, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75,
                    76, 77, 78, 81, 84, 88, 89, 91, 92],
        'Participant11': [2, 3, 10, 11, 12, 24, 33, 40, 45, 46, 48, 50, 51, 52, 54, 56, 72, 76, 78, 89],
        'Participant12': [3, 9, 10, 15, 20, 21, 22, 24, 25, 26, 29, 33, 45, 46, 47, 48, 52, 54, 58, 70, 72, 74, 75, 78,
                          80, 81, 83, 88, 92]}

    mispercievedRan = {
        'Participant1': [5, 7, 12, 14, 16, 19, 20, 26, 27, 28, 29, 31, 32, 33, 35, 44, 45, 46, 47, 50, 55, 58, 64, 65,
                         67, 72, 74, 76, 81, 83, 86, 87, 90, 92, 99, 104, 105, 106, 109, 110, 111, 112, 113, 114, 116,
                         117],
        'Participant2': [8, 13, 15, 17, 19, 21, 24, 26, 27, 28, 29, 32, 35, 39, 44, 45, 46, 47, 49, 54, 58, 59, 64, 65,
                         67, 74, 76, 77, 78, 82, 86, 88, 90, 91, 93, 99, 100, 105, 106, 109, 112, 113, 116],
        'Participant3': [7, 9, 12, 13, 14, 15, 19, 21, 24, 26, 27, 28, 31, 32, 33, 39, 43, 44, 45, 47, 50, 55, 57, 58,
                         59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 74, 76, 81, 85, 86, 88, 89, 93, 95, 96, 99, 100, 103,
                         105, 106, 109, 110, 111, 114, 115, 117],
        'Participant4': [1, 12, 13, 14, 15, 17, 18, 19, 21, 26, 27, 28, 29, 33, 34, 35, 40, 44, 45, 46, 47, 52, 54, 58,
                         59, 64, 65, 66, 72, 76, 80, 81, 83, 87, 88, 92, 93, 95, 99, 101, 104, 105, 106, 108, 109, 110,
                         111, 113, 114],
        'Participant5': [1, 4, 5, 12, 13, 16, 19, 21, 26, 27, 31, 32, 33, 35, 44, 45, 46, 47, 50, 54, 57, 58, 59, 64,
                         65, 66, 67, 69, 76, 81, 86, 88, 89, 90, 93, 94, 99, 101, 104, 105, 106, 107, 109, 110, 112,
                         114, 116],
        'Participant6': [4, 5, 7, 8, 15, 19, 21, 26, 28, 29, 31, 32, 33, 35, 38, 45, 47, 49, 50, 55, 58, 61, 63, 64, 65,
                         66, 67, 68, 74, 76, 81, 82, 88, 89, 90, 93, 95, 98, 99, 103, 104, 105, 106, 109, 110, 112, 115,
                         116],
        'Participant7': [7, 8, 14, 19, 20, 21, 22, 26, 28, 32, 33, 35, 40, 44, 45, 46, 47, 50, 52, 54, 56, 58, 59, 64,
                         65, 67, 72, 76, 79, 80, 81, 83, 90, 93, 94, 99, 100, 101, 104, 105, 106, 107, 108, 109, 110,
                         111, 112, 113, 114, 116],
        'Participant8': [1, 4, 5, 13, 14, 16, 19, 25, 26, 27, 28, 32, 33, 39, 40, 45, 46, 47, 50, 54, 55, 57, 58, 64,
                         67, 76, 81, 83, 85, 86, 88, 89, 93, 95, 99, 102, 105, 106, 107, 109, 110, 114, 116],
        'Participant9': [2, 6, 9, 13, 19, 21, 26, 28, 29, 31, 32, 38, 44, 45, 46, 47, 50, 54, 57, 58, 59, 63, 65, 66,
                         67, 71, 76, 79, 87, 89, 90, 94, 95, 98, 99, 102, 104, 105, 109, 110, 112, 113, 115, 117],
        'Participant10': [6, 8, 9, 10, 13, 14, 16, 17, 19, 21, 25, 26, 27, 28, 29, 31, 32, 33, 43, 45, 46, 47, 48, 54,
                          57, 58, 59, 63, 64, 65, 67, 69, 71, 74, 78, 79, 86, 89, 90, 93, 94, 98, 99, 101, 102, 104,
                          105, 108, 109, 113, 115, 117],
        'Tester1': [6, 7, 8, 9, 10, 14, 15, 17, 19, 21, 25, 26, 27, 28, 29, 32, 33, 34, 35, 38, 44, 45, 47, 50, 57, 58,
                    59, 60, 63, 64, 65, 66, 67, 68, 72, 76, 79, 81, 83, 88, 92, 94, 98, 99, 102, 103, 104, 105, 106,
                    107, 110, 111, 112, 113, 114, 115, 116, 117],
        'Tester2': [5, 8, 9, 13, 14, 15, 16, 17, 19, 21, 26, 27, 28, 29, 31, 32, 34, 38, 39, 44, 45, 46, 47, 50, 52, 54,
                    55, 57, 58, 59, 64, 65, 69, 71, 76, 79, 81, 93, 98, 99, 101, 103, 104, 105, 106, 108, 109, 110, 111,
                    112, 116],
        'Tester3': [1, 4, 5, 6, 13, 14, 19, 21, 26, 27, 28, 29, 31, 32, 33, 35, 38, 45, 46, 47, 50, 52, 54, 55, 57, 58,
                    59, 64, 65, 71, 74, 76, 83, 85, 88, 89, 93, 94, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110,
                    112, 114, 115, 116, 117],
        'Tester4': [1, 2, 4, 5, 7, 8, 12, 19, 21, 25, 26, 27, 28, 31, 32, 33, 36, 38, 44, 45, 46, 47, 50, 54, 55, 57,
                    58, 59, 65, 69, 71, 76, 79, 86, 87, 88, 89, 90, 94, 99, 100, 102, 105, 106, 107, 109, 110, 112, 114,
                    115],
        'Tester5': [1, 2, 5, 7, 14, 16, 19, 26, 28, 29, 32, 35, 39, 44, 45, 46, 47, 50, 55, 58, 59, 64, 65, 71, 81, 93,
                    94, 95, 99, 104, 105, 109, 111, 116, 117],
        'Tester6': [6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37,
                    38, 39, 41, 43, 45, 46, 47, 48, 50, 55, 57, 60, 65, 66, 67, 70, 72, 76, 78, 79, 80, 81, 83, 85, 86,
                    87, 89, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 112, 113, 114,
                    115, 116],
        'Participant11': [4, 5, 6, 10, 12, 14, 22, 26, 32, 33, 43, 45, 46, 54, 55, 56, 58, 65, 68, 76, 78, 79, 81, 83,
                          86, 87, 88, 90, 93, 94, 95, 98, 99, 101, 102, 103, 105, 106, 107, 109, 110, 112, 113, 114,
                          115, 116],
        'Participant12': [2, 13, 14, 16, 19, 21, 27, 28, 29, 31, 32, 43, 45, 46, 47, 52, 54, 55, 56, 58, 64, 71, 79, 80,
                          81, 83, 90, 94, 99, 101, 104, 107, 109, 112, 113, 114, 116]}

    #LOODataSet(18, mispercievedSeq, mispercievedRan)
    keys = list(mispercievedRan.keys())
    for i in range(len(keys)):
        print("Tester{} has perception rate on Random of {}".format(i+1, 1-len(mispercievedRan[keys[i]])/117))

    print("==================================")

    keys = list(mispercievedSeq.keys())
    for i in range(len(keys)):
        print("Tester{} has perception rate on Seq of {}".format(i + 1, 1-len(mispercievedSeq[keys[i]]) / 117))


