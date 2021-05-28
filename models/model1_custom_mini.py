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
                model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
                model.add(Activation('relu')) 
                
                model.add(Activation('relu'))
                model.add(Conv2D(32,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(3,3)))
                model.add(Dropout(0.2))
                model.add(Conv2D(32,(3,3),padding='same'))
                model.add(Activation('relu')) 
                model.add(Conv2D(32,(3,3),padding='same'))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Dropout(0.2))
                model.add(Flatten())
                
                model.add(Activation('relu'))
                model.add(Dense(64))
                model.add(Activation('relu'))
		
                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                return model
                
