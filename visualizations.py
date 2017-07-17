import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import operator
from collections import OrderedDict

import math

def getCsvFootprint(csvfile):
	dataset = pd.read_csv(csvfile, index_col=0, sep=',', dtype='unicode')
	return(sys.getsizeof(dataset))


B_TO_MB_CONVERSION = 1/1048576

def plotFilter():
	GTD_FULL_Size = getCsvFootprint('data/GTD_FULL.csv')
	GTD_FULL_FILTERED_Size = getCsvFootprint('data/GTD_FULL_FILTERED.csv')
	GTD_KNOWNATTACKS_Size = getCsvFootprint('data/GTD_FULL_KNOWNATTACKS.csv')
	GTD_KNOWNATTACKS_FILTERED_Size = getCsvFootprint('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv')
	GTD_UNKNOWNATTACKS_Size = getCsvFootprint('data/GTD_FULL_UNKNOWNATTACKS.csv')
	GTD_UNKNOWNATTACKS_FILTERED_Size = getCsvFootprint('data/GTD_FULL_UNKNOWNATTACKS_FILTERED.csv')

	# data to plot
	# n_groups = 4
	n_groups = 3
	unfiltered_sizes = (GTD_FULL_Size * B_TO_MB_CONVERSION, GTD_KNOWNATTACKS_Size *B_TO_MB_CONVERSION, GTD_UNKNOWNATTACKS_Size*B_TO_MB_CONVERSION)
	filtered_sizes = (GTD_FULL_FILTERED_Size *B_TO_MB_CONVERSION, GTD_KNOWNATTACKS_FILTERED_Size*B_TO_MB_CONVERSION, GTD_UNKNOWNATTACKS_FILTERED_Size*B_TO_MB_CONVERSION)

	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, unfiltered_sizes, bar_width,
					 alpha=opacity,
					 color='b',
					 label='All Fields')

	rects2 = plt.bar(index + bar_width, filtered_sizes, bar_width,
					 alpha=opacity,
					 color='g',
					 label='Numeric Fields')

	plt.xlabel('Dataset')
	plt.ylabel('Size (MB)')
	plt.title('Cleaned vs Dirty Datasets in Memory')
	plt.xticks(index + bar_width, ('Complete', 'Training', 'Prediction'))
	plt.legend()

	plt.tight_layout()
	plt.show()

def clusterGnames(csvFile):
	groupCount ={}
	dataset = pd.read_csv(csvFile, index_col=0, sep=',', dtype='unicode')
	# gnames = dataset.gname
	# for group in gnames:
	for index,row in dataset.iterrows():
		group = row['gname']
		if not row['nkill'] or math.isnan(float(row['nkill'])):
			kills = 0
		else:
			kills = int(row['nkill'])
		# print(group)
		# print(row['nkill'])
		if group in groupCount:
			groupCount[group]['count']+=1
			# print(groupCount[group])
			groupCount[group]['kills'] += kills
		else:
			groupCount[group] ={'kills':0}
			groupCount[group]['count'] = 1
			groupCount[group]['kills'] += kills

	# print(groupCount)
	# groupCount = sorted(groupCount.items(),key=operator.itemgetter(1), reverse=True)[0:29]
	#Extracts top 30
	groupCount = OrderedDict(sorted(groupCount.items(), key=lambda i: i[1]['count'], reverse=True)[0:29])
	# groupCount = OrderedDict(sorted(groupCount.items(), key=lambda i: i[1]['count'], reverse=True))

	print(groupCount)

	colors ={}
	x=[]
	y=[]
	for group in groupCount.items():
		print(group)
		# COUNT_NDX = 1
		# KILLS_NDX = 0
		x.append(group[1]['count'])
		y.append(group[1]['kills'])
	plt.scatter(x,y)
	plt.xlabel('Incidents')
	plt.ylabel('Kills')
	plt.show()

def scatterKillInjured(csvFile, title, head):
	groupCount ={}
	dataset = pd.read_csv(csvFile, index_col=0, sep=',', dtype='unicode')
	colors ={}
	x=[]
	y=[]
	for index,row in dataset.iterrows():
		if row['nwound'] and  not math.isnan(float(row['nwound'])):
			x.append(int(row['nwound']))
		else:
			x.append(0)
		if row['nkill'] and not math.isnan(float(row['nkill'])):
			y.append(int(row['nkill']))
		else:
			y.append(0)
	# print(sorted(list(zip(x,y)),key=lambda tup: tup[1], reverse=True)[0:30])
	# print(sorted(list(zip(x,y)),key=lambda tup: tup[0] +tup[1], reverse=True)[0:head])
	t = sorted(list(zip(x,y)),key=lambda tup: tup[0] +tup[1], reverse=True)[0:head]
	x = list(zip(*t))[0]
	y = list(zip(*t))[1]
	fig = plt.figure()
	plt.scatter(x,y)
	plt.title(title)
	plt.xlabel('Wounded')
	plt.ylabel('Kills')
	plt.show()

# plotFilter()
# clusterGnames('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv')
scatterKillInjured('data/GTD_FULL_UNKNOWNATTACKS_FILTERED.csv', title="50 Unknown Attacks",head=100)
scatterKillInjured('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv', title="50 Known Attacks",head=100)
# scatterKillInjured('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv', title="Known Attacks")
# scatterKillInjured('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv', title="Known Attacks")