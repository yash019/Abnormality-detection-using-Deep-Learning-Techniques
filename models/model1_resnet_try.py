from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k 
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D

class tinyVGG:
	@staticmethod
	def build(height,width,depth,classes):
		model=Sequential()
		input_shape=(height,width,depth)
		channel_dim = -1

		if(k.image_data_format() == 'channels_first'):
			input_shape = (depth,width,height)
			channel_dim = 1
                model.add(Conv2D(64,(7,7),padding='same',input_shape=input_shape))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))

                model.add(MaxPooling2D(pool_size=(3,3)))		

                #model.add(Dropout(0.25))

                #shortcut(with a conv)
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                        #add

                        #shortcut(without a conv)
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                        #add

                model.add(MaxPooling2D(pool_size=(2,2)))

                        #model.add(Dropout(0.25))

                        #shortcut(with a conv)
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                        #add

                        #shortcut(without a conv)
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(64,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                        #add

                model.add(MaxPooling2D(pool_size=(2,2)))
                        #model.add(Dropout(0.25))


                        #shortcut 
                model.add(Conv2D(128,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))

                        #model.add(Conv2D(128,(3,3),padding='same'))

                model.add(Conv2D(128,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                model.add(Conv2D(128,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(BatchNormalization(axis=channel_dim))
                        #add

                        #(+) relu or at the end

                keras.layers.GlobalMaxPooling2D(data_format=None))

                        #model.add(Dropout(0.25))

                model.add(Flatten())
                model.add(Dense(1024))
                model.add(Activation('relu')) 
                model.add(BatchNormalization()) 
                #model.add(Dropout(0.5))
                model.add(Dense(3))
                model.add(Activation('sigmoid'))

		return model


#__init__ -> constructor in python
#from folder.filename import classname
	

