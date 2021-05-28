from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k 
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D

#C + MP + DO + (2C + MP + DO)X2 + FLATTEN +DENSE(RELU) + BN + DO + DENSE(SOFTMAX)
#c = Conv2D + ReLU + BN
class tinyVGG:
	@staticmethod
	def build(height,width,depth,classes):
		model=Sequential()
		input_shape=(height,width,depth)
		channel_dim = -1

		if(k.image_data_format() == 'channels_first'):
			input_shape = (depth,width,height)
			channel_dim = 1
		model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.25))

# Increase Filters and decrease pool size

		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))


		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))


		model.add(Conv2D(64,(3,3),padding='same'))  #more features extracted
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(Conv2D(32,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=channel_dim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))


		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(BatchNormalization())  #already flattened into 1D data,the axis value is 0
		model.add(Dropout(0.5))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		return model


#__init__ -> constructor in python
#from folder.filename import classname
	

