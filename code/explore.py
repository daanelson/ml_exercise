import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns			#only here to make graphs look nicer.
import utils
import sys
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.cross_validation import train_test_split
from sklearn import metrics



if __name__ == '__main__':
	#file to read as input argument
	rawdata = pd.read_csv(sys.argv[1],error_bad_lines = False)

	#values to remove based on initial inspection of the data (r.describe, pandas.summary())
	handRemoved = ['STATE','ZIP','CONTROLN']
	rawdata = rawdata.drop(handRemoved,axis=1)

	#binarize resulting data (all columns with "object" datatype)
	bindata = pd.get_dummies(rawdata,prefix = 'BIN')			

	#normalizes all remaining non-binary features, fills N/A data w/mean
	bindata = utils.normalize(bindata)
	bindata = bindata.fillna(0)

	#based on ANOVA test with TARGET_B, removes all but val best features. Use this as sweep
	vals = np.arange(200,525,25)
	for val in vals:
		_, featdata = utils.bestfeat(bindata,val)

		#split into training/test 
		feat_train, feat_test, b_train, d_train, b_test, d_test = utils.trainTest(featdata,.30)

		#logistic regression
		classWeights = {0:1,1:20}
		clf = LogisticRegression(class_weight = classWeights, penalty = 'l1')
		clf.fit(feat_train, b_train)

		#scoring - predicted $ vs. actual $ from test sample as % and 
		predictions = clf.predict(feat_test)

		#sum of all non-zero prediction actual donations minus cost * number of nonzero predictions
		predCash = np.sum(np.multiply(predictions,d_test)) - 0.68*(np.sum(predictions))
		normCash = np.sum(d_test) - 0.68*len(d_test)
		possCash = np.sum(d_test) - 0.68*(np.sum(b_test))
		print "NumFeats {0}".format(val)
		print "Actual revenue: {0}, Classifier revenue: {1}, Possible revenue: {2}".format(normCash, predCash, possCash)
		print "Positive Predictions: {0} Accurate Predictions: {3} Total positives: {1} Total Possible: {2}".format(sum(predictions),sum(b_test),len(b_test), sum(np.multiply(predictions,b_test)))

		#accuracy - not that useful but easy to calculate
		accScore = clf.score(feat_test,b_test)
		print "Accuracy score : ", accScore

		#AUC
		preds = clf.predict_proba(feat_test)[:,1]

		#auc curve, wants probability estimates of positive class. Uncomment to view curves
		fpr, tpr, _ = metrics.roc_curve(b_test, preds)
		print 'AUC:', metrics.auc(fpr, tpr)
		# plt.plot(fpr, tpr, lw=1)
		# plt.ylabel('fpr')
		# plt.xlabel('tpr')
		# plt.title('AUC for LogisticRegression')
		# plt.show()
		# plt.savefig("AUCgraphwithfeats{0}.png".format(val))



