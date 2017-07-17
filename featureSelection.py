# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "data/GTD_FULL_KNOWNATTACKS_NOPGIS.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv(url)
array = dataframe.values
gnameNdx = list(dataframe).index('gname')
print("Gname index: %i\nEnding Index %i" %(gnameNdx, len(list(dataframe))-1))
# print(array)
X = array[:,0:136]
numpy.delete(X,60,axis=1)
Y = array[:,60]
print(X)
print(Y)
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])