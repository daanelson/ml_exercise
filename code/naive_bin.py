import pandas as pd
import numpy as np
import sys
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == '__main__':
	#read in raw data as pandas dataframe
	bindata = pd.read_csv('binnednormed.csv', error_bad_lines = False, warn_bad_lines = True)
	#going from cleaning to classifying. 

	#lets us train for both
	labels = zip(bindata['TARGET_B'],bindata['TARGET_D'])

	features = bindata.drop(['TARGET_B','TARGET_D','CONTROLN'],axis=1)

	feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size = 0.30, random_state = 22)

	b_train,d_train = zip(*label_train)
	b_test,d_test = zip(*label_test)

	#LogisticRegression for 'Target_B'. Weighting heavily b/c have less positive v. negative data
	clf = RandomForestClassifier(class_weight={0:1,1:200}, n_jobs=-1)
	clf.fit(feat_train,b_train)
	preds = clf.predict_proba(feat_test)[:,1]

	#auc curve, wants probability estimates of positive class.
	fpr, tpr, _ = metrics.roc_curve(b_test, preds)
	print 'AUC:', metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=1)
	plt.ylabel('fpr')
	plt.xlabel('tpr')
	plt.title('AUC Curve for LogisticRegression')
	plt.show()
    #plt.savefig("AUCgraphOfLog.jpg".format(plotNum))