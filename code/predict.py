import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import sys
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.cross_validation import train_test_split
from sklearn import metrics

if __name__ == '__main__':

	#file to read as input argument
	rawdata = pd.read_csv(sys.argv[1],error_bad_lines = False)

	#values to remove based on initial inspection of the data
	handRemoved = ['STATE','ZIP','CONTROLN']
	rawdata = rawdata.drop(handRemoved,axis=1)

	#binarize resulting data (all columns with "object" datatype)
	bindata = pd.get_dummies(rawdata,prefix_sep = 'BIN')			

	#normalizes all remaining non-binary features, fills N/A data w/mean
	featdata = utils.normalize(bindata)
	bindata = bindata.fillna(0)

	#based on anova test with TARGET_B, removes all but 400 best features. Return kbest as well for use with test data
	kbest, featdata = utils.bestfeat(bindata,400)

	#split into training/test
	feat_train, feat_test, b_train, d_train, b_test, d_test = utils.trainTest(featdata,.30)

	#logistic regression
	classWeights = {0:1,1:20}
	clf = LogisticRegression(class_weight = classWeights, penalty = 'l1')
	feat_train = feat_train.drop(['OSOURCEBINCLL','OSOURCEBINPTP','RFA_3BINA2C','RFA_6BINU1C','RFA_10BINA2B'],axis=1)	#overfitting, perhaps? These were not present in test data
	clf.fit(feat_train, b_train)

	#prepping prediction data using same steps as training data
	predata = pd.read_csv(sys.argv[2],error_bad_lines = False)
	controln = predata['CONTROLN']
	predata = predata.drop(handRemoved,axis=1)
	binpred = pd.get_dummies(predata,prefix_sep = 'BIN')
	binpred = utils.normalize(binpred)
	predfeats = binpred.fillna(0)
	predcols = bindata.drop(['TARGET_B','TARGET_D'],axis=1).columns[kbest.get_support(True)]
	predcols = predcols.drop(['OSOURCEBINCLL','OSOURCEBINPTP','RFA_3BINA2C','RFA_6BINU1C','RFA_10BINA2B'])
	predfeats = predfeats[predcols]

	preds = clf.predict(predfeats)

	pd.Series(data=preds,index=controln).to_csv('predictions.csv')
	

