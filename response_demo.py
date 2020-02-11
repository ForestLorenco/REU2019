import pandas as pd
import collections

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

def video_seperate_gender(cls):
	f_response = 0
	f_total = 0
	m_response = 0
	m_total = 0

	f_emot_dic = defaultdict(lambda: [0,0])
	m_emot_dic = defaultdict(lambda: [0,0])
	for i in range(1,19):
		tester = "Tester"+str(i)
		filename = tester+".csv"
		df = pd.read_csv("ResponseData"+cls+"/Response"+filename)
		df = df.values

		correct = pd.read_csv("ResponseData"+cls+"/CorrectResponse.csv")
		correct = correct.values
		

		ran = pd.read_csv("Random_Video_Analytics.csv")
		ran = ran.values

		seq = pd.read_csv("Seq_Video_Analytics.csv")
		seq = seq.values
		#print(len(ran), len(df))
		
		if cls == "Random":
			for i in range(2,len(df)):
				if ran[i-2][0] == "Male":
					m_total += 1
					if correct[i] == df[i]:
						m_response += 1
						if correct[i] == 1:
							m_emot_dic["Positive"][0]+=1
						elif correct[i] == 3:
							m_emot_dic["Neutral"][0]+=1
						else:
							m_emot_dic["Negative"][0]+=1

					if correct[i] == 1:
						m_emot_dic["Positive"][1]+=1
					elif correct[i] == 3:
						m_emot_dic["Neutral"][1]+=1
					else:
						m_emot_dic["Negative"][1]+=1
				else:
					f_total += 1
					if correct[i] == df[i]:
						f_response += 1
						if correct[i] == 1:
							f_emot_dic["Positive"][0]+=1
						elif correct[i] == 3:
							f_emot_dic["Neutral"][0]+=1
						else:
							f_emot_dic["Negative"][0]+=1
							
					if correct[i] == 1:
						f_emot_dic["Positive"][1]+=1
					elif correct[i] == 3:
						f_emot_dic["Neutral"][1]+=1
					else:
						f_emot_dic["Negative"][1]+=1

			
	print("Male video response: {} | Female video response {}".format(m_response/m_total, f_response/f_total))
	print()
	print("Male ")
	print("Negative: {}".format(m_emot_dic["Negative"][0]/m_emot_dic["Negative"][1]))
	print("Positive: {}".format(m_emot_dic["Positive"][0]/m_emot_dic["Positive"][1]))
	print("Neutral: {}".format(m_emot_dic["Neutral"][0]/m_emot_dic["Neutral"][1]))

	print("Female ")
	print("Negative: {}".format(f_emot_dic["Negative"][0]/f_emot_dic["Negative"][1]))
	print("Positive: {}".format(f_emot_dic["Positive"][0]/f_emot_dic["Positive"][1]))
	print("Neutral: {}".format(f_emot_dic["Neutral"][0]/f_emot_dic["Neutral"][1]))

def tester_seperate_gender(cls):
	f_response = 0
	f_total = 0
	m_response = 0
	m_total = 0

	f_emot_dic = defaultdict(lambda: [0,0])
	m_emot_dic = defaultdict(lambda: [0,0])

	for i in range(1,19):
		tester = "Tester"+str(i)
		df = pd.read_csv("ResponseData"+cls+"/Response"+tester+".csv")
		df = df.values

		correct = pd.read_csv("ResponseData"+cls+"/CorrectResponse.csv")
		correct = correct.values
		

		ran = pd.read_csv("Random_Video_Analytics.csv")
		ran = ran.values

		seq = pd.read_csv("Seq_Video_Analytics.csv")
		seq = seq.values
		#print(len(ran), len(df))
		
		t = pd.read_csv("TesterInformation.csv")
		t = t.values
		#print(t)
		index = i-1
		for i in range(2,len(df)):
			if "Male" == t[index][2]:
				m_total += 1
				if correct[i] == df[i]:
					m_response += 1
					if correct[i] == 1:
						m_emot_dic["Positive"][0]+=1
					elif correct[i] == 3:
						m_emot_dic["Neutral"][0]+=1
					else:
						m_emot_dic["Negative"][0]+=1

				if correct[i] == 1:
					m_emot_dic["Positive"][1]+=1
				elif correct[i] == 3:
					m_emot_dic["Neutral"][1]+=1
				else:
					m_emot_dic["Negative"][1]+=1
			else:
				f_total += 1
				if correct[i] == df[i]:
					f_response += 1
					if correct[i] == 1:
						f_emot_dic["Positive"][0]+=1
					elif correct[i] == 3:
						f_emot_dic["Neutral"][0]+=1
					else:
						f_emot_dic["Negative"][0]+=1
							
				if correct[i] == 1:
					f_emot_dic["Positive"][1]+=1
				elif correct[i] == 3:
					f_emot_dic["Neutral"][1]+=1
				else:
					f_emot_dic["Negative"][1]+=1

	print("Male tester response: {} | Female video response {}".format(m_response/m_total, f_response/f_total))
	print()
	print("Male ")
	print("Negative: {}".format(m_emot_dic["Negative"][0]/m_emot_dic["Negative"][1]))
	print("Positive: {}".format(m_emot_dic["Positive"][0]/m_emot_dic["Positive"][1]))
	print("Neutral: {}".format(m_emot_dic["Neutral"][0]/m_emot_dic["Neutral"][1]))

	print("Female ")
	print("Negative: {}".format(f_emot_dic["Negative"][0]/f_emot_dic["Negative"][1]))
	print("Positive: {}".format(f_emot_dic["Positive"][0]/f_emot_dic["Positive"][1]))
	print("Neutral: {}".format(f_emot_dic["Neutral"][0]/f_emot_dic["Neutral"][1]))
	
def video_seperate_race(cls):
	d = defaultdict(lambda: [0,0])

	emot_dic = defaultdict(lambda: defaultdict(lambda: [0,0]))

	for i in range(1,19):
		tester = "Tester"+str(i)
		filename = tester+".csv"

		df = pd.read_csv("ResponseData"+cls+"/Response"+filename)
		df = df.values

		correct = pd.read_csv("ResponseData"+cls+"/CorrectResponse.csv")
		correct = correct.values

		ran = pd.read_csv("Random_Video_Analytics.csv")
		ran = ran.values


		if cls == "Random":
			for i in range(2,len(df)):
				d[ran[i-2][1]][1] += 1
				if correct[i] == df[i]:
					d[ran[i-2][1]][0] += 1
					if correct[i] == 1:
						emot_dic[ran[i-2][1]]["Positive"][0]+=1
					elif correct[i] == 3:
						emot_dic[ran[i-2][1]]["Neutral"][0]+=1
					else:
						emot_dic[ran[i-2][1]]["Negative"][0]+=1

				if correct[i] == 1:
					emot_dic[ran[i-2][1]]["Positive"][1]+=1
				elif correct[i] == 3:
					emot_dic[ran[i-2][1]]["Neutral"][1]+=1
				else:
					emot_dic[ran[i-2][1]]["Negative"][1]+=1



			
	for k,v in d.items():
		print("Response for video race {} is {}".format(k, v[0]/v[1]))
	print()

	for k,v in emot_dic.items():
		print(k)
		for k_e, v_e in v.items():
			print("{}: {}".format(k_e, v_e[0]/v_e[1]))
	print()



def tester_seperate_race(cls):
	d = defaultdict(lambda: [0,0])

	emot_dic = defaultdict(lambda: defaultdict(lambda: [0,0]))
	for i in range(1,19):
		tester = "Tester"+str(i)
		df = pd.read_csv("ResponseData"+cls+"/Response"+tester+".csv")
		df = df.values

		correct = pd.read_csv("ResponseData"+cls+"/CorrectResponse.csv")
		correct = correct.values
		

		ran = pd.read_csv("Random_Video_Analytics.csv")
		ran = ran.values

		seq = pd.read_csv("Seq_Video_Analytics.csv")
		seq = seq.values
		#print(len(ran), len(df))
		
		t = pd.read_csv("TesterInformation.csv")
		t = t.values
		#print(t)
		index = i-1
		for i in range(2,len(df)):
			d[t[index][3]][1]+=1
			if correct[i] == df[i]:
				d[t[index][3]][0] += 1
				if correct[i] == 1:
					emot_dic[t[index][3]]["Positive"][0]+=1
				elif correct[i] == 3:
					emot_dic[t[index][3]]["Neutral"][0]+=1
				else:
					emot_dic[t[index][3]]["Negative"][0]+=1

			if correct[i] == 1:
				emot_dic[t[index][3]]["Positive"][1]+=1
			elif correct[i] == 3:
				emot_dic[t[index][3]]["Neutral"][1]+=1
			else:
				emot_dic[t[index][3]]["Negative"][1]+=1
		
	for k,v in d.items():
		print("Response for tester race {} is {}".format(k, v[0]/v[1]))
	print()

	for k,v in emot_dic.items():
		print(k)
		for k_e, v_e in v.items():
			print("{}: {}".format(k_e, v_e[0]/v_e[1]))
	print()
if __name__ == "__main__":
	#video_seperate_gender("Random")

	#tester_seperate_gender("Random")

	#video_seperate_race("Random")
	
	tester_seperate_race("Random")