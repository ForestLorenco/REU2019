fileR = open("ResponseDataModels/RandomPred.csv","w")
fileC = open("ResponseDataModels/RandomCorrect.csv","w")

#1 - Pos, 2 - Neg, 3- Neu
for i in range( 796):
	fileR.write("1\n")
	fileC.write("1\n")

for i in range(620):
	fileR.write("2\n")
	fileC.write("2\n")

for i in range( 705):
	fileR.write("3\n")
	fileC.write("3\n")

#Positive but not classified right
for i in range(172):
	fileR.write("2\n")
	fileC.write("1\n")

for i in range(166):
	fileR.write("3\n")
	fileC.write("1\n")

#Negative but not classified right
for i in range(263):
	fileR.write("1\n")
	fileC.write("2\n")

for i in range(251):
	fileR.write("3\n")
	fileC.write("2\n")

#Neutral but not classified right
for i in range(219):
	fileR.write("1\n")
	fileC.write("3\n")

for i in range(210):
	fileR.write("2\n")
	fileC.write("3\n")


