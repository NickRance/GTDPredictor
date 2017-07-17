import pandas as pd
from pprint import pprint
import config

def generateKnownAttacks():
	dataset = pd.read_csv('data/GTD_FULL.csv',index_col=0, sep=',',	 dtype='unicode')
	dataset = dataset[dataset.gname != "Unknown"]
	dataset.to_csv("data/GTD_FULL_KNOWNATTACKS.csv", sep=',', encoding='utf-8')

def generateUnknownAttacks():
	dataset = pd.read_csv('data/GTD_FULL.csv',index_col=0, sep=',', dtype='unicode')
	dataset = dataset[dataset.gname == "Unknown"]
	dataset.to_csv("data/GTD_FULL_UNKNOWNATTACKS.csv", sep=',', encoding='utf-8')

def main():
	dataset = pd.read_csv('data/GTD_FULL_KNOWN.csv',index_col=0, sep=',', dtype='unicode')

def extractPGIS(csvfile):
	"""
	:param csvfile: GTD CSV filepath containing records before and after 1998 when PGIS was responsible for data collection
	:return: Filename of clone of input csvfile without the terrorist attacks before 1998
	"""
	dataset = pd.read_csv(csvfile, sep=',',index_col=0, dtype='unicode')
	dataset.iyear = dataset.iyear.astype(int)
	dataset = dataset[dataset.iyear >= 1998]
	outputFilePath = csvfile.replace(".csv","") + "_NOPGIS" + ".csv"
	dataset.to_csv(outputFilePath, sep=',', encoding='utf-8')
	return outputFilePath

def filterStrings(csvfile):
	#Axis = 1  is Columns
	filterFields = config.getFilteredFields()
	dataset = pd.read_csv(csvfile, sep=',',index_col=0, dtype='unicode')
	for field in filterFields:
		dataset.drop(field, axis=1, inplace=True)
	outputFilePath = csvfile.replace(".csv","") + "_FILTERED" + ".csv"
	dataset.to_csv(outputFilePath, sep=',', encoding='utf-8')

def codifyGname(csvfile):
	dataset = pd.read_csv(csvfile, index_col=0, sep=',', dtype='unicode')
	dataset = dataset['gname']
	print(dataset)

# codifyGname('data/GTD_FULL.csv')

# main()
# generateUnknownAttacks()
# generateKnownAttacks()
# extractPGIS(csvfile="data/GTD_FULL_KNOWNATTACKS.csv")
filterStrings(csvfile="data/GTD_FULL_KNOWNATTACKS.csv")
filterStrings(csvfile="data/GTD_FULL_UNKNOWNATTACKS.csv")
# print(sorted(config.getFieldList()))
# print(config.getTextFields())
# print([item for item in config.getFieldList() if item not in config.getTextFields()])
