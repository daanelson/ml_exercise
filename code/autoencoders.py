import sys
import pandas as pd
import numpy as np
from keras.layers import containers 
from keras.layers.core import AutoEncoder, Dense
from keras.optimizers import RMSprop
from keras.models import Sequential


if __name__ == '__main__':
	#reads in csv - right now only testing with pre-binarized features
	features = pd.read_csv(sys.argv[1])
	features = features.drop(['TARGET_B','TARGET_D'],axis=1)
	features = features.values.astype(np.float32)
	numfeats = len(features[0])
	batch_size = 300
	nb_epoch = 5 

	encoder = containers.Sequential([Dense(400, activation = 'sigmoid', input_dim = numfeats), Dense(400, activation = 'sigmoid')])
	decoder = containers.Sequential([Dense(400, input_dim = 400), Dense(numfeats)])

	autoencoder = Sequential()
	autoencoder.add(AutoEncoder(encoder = encoder, decoder = decoder, output_reconstruction = False))
	autoencoder.compile(loss = 'mean_squared_error', optimizer = RMSprop())

	fittinghist = autoencoder.fit(features, features, batch_size = batch_size, nb_epoch = nb_epoch)
	predata = autoencoder.predict(features)