
import pandas as pd
import sys
import numpy as np
import utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle	
from explore import variance_thresh
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2

#reads data, does some preprocessing for the first time, saves to csv
def process_data(point):
	bindata = pd.read_csv(point,error_bad_lines=False)
	bindata = bindata.drop(['STATE','ZIP','CONTROLN'],axis=1)

	bindata = pd.get_dummies(bindata, prefix = 'BIN')
	#may need to fillNA
	bindata = utils.normalize(bindata)

	labels = zip(bindata['TARGET_B'],bindata['TARGET_D'])
	features = bindata.drop(['TARGET_B','TARGET_D'],axis=1)

	features.fillna(0)
	#return labels.values, features.values
	#removes all categorical variables which are the same OR N/A in 75% of entries. 
	bindata = variance_thresh(bindata,0.8)
	bindata.to_csv('data/binnednn2.csv')

	feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state = 12)
	label_train, d_train = zip(*label_train)
	label_test, d_test = zip(*label_test)

	label_train = [[np.float32(val == 0), np.float32(val == 1)] for val in label_train]
	label_test = [[np.float32(val == 0), np.float32(val == 1)] for val in label_test]
	return feat_train.values.astype(np.float32), feat_test.values.astype(np.float32), label_train, label_test, d_test

#should use on subsequent calls once data has been read
def get_data(ref):
	bindata = pd.read_csv(ref)
	labels = zip(bindata['TARGET_B'],bindata['TARGET_D'])
	features = bindata.drop(['TARGET_B','TARGET_D'],axis=1)
	features.fillna(0)

	#gives us train/test split on data
	feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state = 12)
	label_train, d_train = zip(*label_train)
	label_test, d_test = zip(*label_test)

	label_train = [[np.float32(val == 0), np.float32(val == 1)] for val in label_train]
	label_test = [[np.float32(val == 0), np.float32(val == 1)] for val in label_test]
	return feat_train.values.astype(np.float32), feat_test.values.astype(np.float32), label_train, label_test, d_test


if __name__ == '__main__':
	#can vary, needs to be high so as to not train a trivial predictor
	weight_cost = 40

	#call process_data the first time, get_data afterwards so as to not go through
	X_train, X_test, y_train, y_test, d_test = process_data(sys.argv[1])
	numfeats = len(X_train[0])
	nb_epoch = 5
	batch_size = int(len(X_train)/100)

	#weights positive samples by weight cost, 15-20 gives better results
	y_weight = [1 + val[1]*weight_cost for val in y_train]			
	y_weight = np.array(y_weight)

	#builds up model layer by layer
	model = Sequential()
	model.add(Dense(int(numfeats/2), activation = 'sigmoid',input_shape=(numfeats,)))
	model.add(Dropout(0.9))
	model.add(Dense(int(numfeats/2), activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation = 'softmax'))
	model.compile(loss='hinge', optimizer='adam')

	#trains and scores model
	fittinghist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1, sample_weight = y_weight)
	score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	predpos = model.predict_classes(X_test)
	realpos = [val[1] for val in y_test]

	#sum of all non-zero prediction actual donations minus cost * number of nonzero predictions
	predCash = np.sum(np.multiply(predpos,d_test)) - 0.68*(np.sum(predpos))
	normCash = np.sum(d_test) - 0.68*len(d_test)
	possCash = np.sum(d_test) - 0.68*(np.sum(realpos))

	print "Actual revenue: {0}, Classifier revenue: {1}, Possible revenue: {2}".format(normCash, predCash, possCash)
	print "Positive Predictions: {0} Accurate Predictions: {3} Total positives: {1} Total Possible: {2}".format(sum(predpos),sum(realpos),len(predpos), sum(np.multiply(predpos,realpos)))
