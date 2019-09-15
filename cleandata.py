#Keyboard events messed up for particpants so have to translate them to correct input
def translate(s):
	if s == "End":
		return "1"
	elif s == "Down":
		return "2"
	elif s == "Clear":
		return "5"
	elif s == "Right":
		return "6"
	elif s == "Next":
		return "3"
	elif s == "Left":
		return "4"
	else:
		return s[-1]

def compare(key,response,correct,unmatched,index):
	yes,no = 0,0
	fiar = 0
	for i in range(len(response)):
		if(correct[i]==int(response[i])):
			yes += 1
		else:
			no += 1
			#maybe do i+1 here, for readability 
			index[key].append(i+1)
			unmatched[key].append(correct[i])
	return (yes/len(correct),no/len(correct));

def tally(key, unmatched):
	t = [0]*6
	for element in unmatched:
		t[element-1] += 1
	print("Incorrect Totals for %s | HAP: %d | ANG: %d | NEU: %d | SAD: %d | DIS: %d | FEA: %d" % (key,t[0],t[1],t[2],t[3],t[4],t[5]))

def forConfusion(response, correct,type):
	file = open("ResponseData"+type+"/CorrectResponse.txt","w")
	for i in range(len(correct)):
		file.write(str(correct[i])+"\n")
	file.close()


	for i in range(6):
		file = open("ResponseData"+type+"/ResponseTester"+str(i+1)+".txt","w")
		for element in response["Tester"+str(i+1)]:
			file.write(str(element)+"\n")
		file.close()
	for i in range(12):
		file = open("ResponseData"+type+"/ResponseParticipant"+str(i+1)+".txt","w")
		for element in response["Participant"+str(i+1)]:
			file.write(str(element)+"\n")
		file.close()
		


def main():
	fileRandom = open("EyeTrackingExperimentRandomKeyDataExport.csv","r")
	fileSequential = open("EyeTrackingExperimentSequentialKeyDataExport.csv","r")
	
	correctSeq = [3,2,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,4,2,5,6,1,3,2,5,6,1,3,4]
	correctRan = [2,5,1,2,2,6,2,3,5,6,2,6,1,2,4,3,4,1,6,2,4,3,1,5,5,6,2,4,4,6,5,2,6,2,1,5,1,4,5,3,3,1,3,6,2,5,6,1,3,6,5,3,5,3,1,3,6,3,4,6,3,3,5,2,4,4,4,5,5,1,5,4,1,6,1,2,1,6,4,2,2,3,2,3,3,4,4,6,5,4,1,2,5,6,5,1,1,4,6,4,4,6,2,2,4,4,2,4,2,3,2,4,4,2,6,2,5]

	rdic = {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}
	sdic= {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}
	
	sunmatched = {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}
	runmatched = {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}

	s_indicesUM = {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}
	r_indicesUM = {"Tester1":[],"Tester2":[],"Tester3":[],"Tester4":[],"Tester5":[],"Tester6":[],"Participant1":[],"Participant2":[],"Participant3":[],"Participant4":[],"Participant5":[],"Participant6":[],"Participant7":[],"Participant8":[],"Participant9":[],"Participant10":[],"Participant11":[],"Participant12":[]}

	#Getting key presses and cleaning the data of all none number characters from Random
	done = False
	temp = False
	qcounter = 0
	while not done:
		line = fileRandom.readline()
		if line == "":
			break
		
		response = ""
		tester = 0

		if "VideoStimulusStart" in line:
			line = fileRandom.readline()
			if "KeyboardEvent" in line:
				for i in range(1,13):
					if "Tester"+str(i)+"," in line:
						response = line[len(line)-2]
						tester = i
						rdic["Tester"+str(tester)].append(response)
						temp = True

					if "Participant"+str(i)+"," in line:
						part = line.split(",")
						response = translate(part[-1][:-1])
						tester = i
						rdic["Participant"+str(tester)].append(response)
						temp = True

		if "Question Slide" in line and qcounter == 0:
			qcounter += 1
			line = fileRandom.readline()
			if "KeyboardEvent" in line:
				for i in range(1,13):
					if "Tester"+str(i)+"," in line:
						response = line[len(line)-2]
						tester = i
						if temp:
							rdic["Tester"+str(tester)]=rdic["Tester"+str(tester)][:-1]
							rdic["Tester"+str(tester)].append(response)
							temp = False
						else:
							rdic["Tester"+str(tester)].append(response)
					if "Participant"+str(i)+"," in line:
						part = line.split(",")
						response = translate(part[-1][:-1])
						tester = i
						if temp:
							rdic["Participant"+str(tester)]=rdic["Participant"+str(tester)][:-1]
							rdic["Participant"+str(tester)].append(response)
							temp = False
						else:
							rdic["Participant"+str(tester)].append(response)
			temp = False

		if "Question Slide" in line and qcounter == 1:
			qcounter -= 1

	for key in rdic:
		newlist = []
		for element in rdic[key]:
			if element in ['1','2','3','4','5','6']:
				newlist.append(element)
		rdic[key] = newlist


	#Getting key presses and cleaning the data of all none number characters from Sequential
	done = False
	temp = False
	qcounter = 0
	while not done:
		line = fileSequential.readline()
		if line == "":
			break
		
		response = ""
		tester = 0

		if "VideoStimulusStart" in line:
			line = fileSequential.readline()
			if "KeyboardEvent" in line:

				for i in range(1,13):
					if "Tester"+str(i)+"," in line:
						response = line[len(line)-2]
						tester = i
						sdic["Tester"+str(tester)].append(response)
						temp = True
					if "Participant"+str(i)+"," in line:
						part = line.split(",")
						response = translate(part[-1][:-1])
						tester = i
						sdic["Participant"+str(tester)].append(response)
						temp = True

		if "Question Slide" in line and qcounter == 0:

			qcounter += 1
			line = fileSequential.readline()
			if "KeyboardEvent" in line:
				
				for i in range(1,13):
					##print(line, ("Participant"+str(i) in line))
					if "Tester"+str(i)+"," in line:
						response = line[len(line)-2]
						tester = i
						if temp:
							sdic["Tester"+str(tester)]=sdic["Tester"+str(tester)][:-1]
							sdic["Tester"+str(tester)].append(response)
							temp = False
						else:
							sdic["Tester"+str(tester)].append(response)
					if "Participant"+str(i)+"," in line:
						part = line.split(",")
						response = translate(part[-1][:-1])
						tester = i
						if temp:
							sdic["Participant"+str(tester)]=sdic["Participant"+str(tester)][:-1]
							sdic["Participant"+str(tester)].append(response)
							temp = False
						else:
							sdic["Participant"+str(tester)].append(response)
			temp = False

		if "Question Slide" in line and qcounter == 1:
			qcounter -= 1

	for key in sdic:
		newlist = []
		for element in sdic[key]:
			if element in ['1','2','3','4','5','6']:
				newlist.append(element)
		sdic[key] = newlist

	#Tester6 sucked at life so this fixes it a bit
	sdic["Tester6"].insert(7,"1")
	rdic["Tester6"].insert(7,"1")
	rdic["Participant10"].insert(7,"1")
	rdic["Participant10"].insert(112,"1")
	sdic["Participant1"].insert(0,"1")


	
	#compare to correct Sequential
	for key in sdic:
		print(key + "| Correct percent: %f Incorrect percent: %f" % compare(key,sdic[key],correctSeq,sunmatched,s_indicesUM))
	#print(sunmatched)

	for key in sunmatched:
		tally(key, sunmatched[key])

	#Random
	for key in rdic:
		print(key + "| Correct percent: %f Incorrect percent: %f" % compare(key,rdic[key],correctRan,runmatched,r_indicesUM))

	for key in runmatched:
		tally(key, runmatched[key])

	forConfusion(sdic, correctSeq,"Sequential")
	forConfusion(rdic, correctRan,"Random")
	

	'''
	for key in rdic:
		print(key,len(rdic[key]))
		print(key,len(sdic[key]))
	#print("Correct" ,correctRan,len(correctRan))
	'''
	print(s_indicesUM)
	print()
	print(r_indicesUM)
	fileSequential.close()
	fileRandom.close()
main()