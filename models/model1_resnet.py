#importing libraies
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k 
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D


class resnet:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1
        # To make it independent of the software used, theano or tensorflow
        if (k.image_data_format() == 'channels_first'):
            input_shape = (depth, height, width)
            channel_dim = 1
            
        inputs = Input(shape=input_shape)

        # First convolution block
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',input_shape=input_shape)(inputs)
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # Max Pool 2D will higlight the features that will help model to learn easily 
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Second convolution block
        # Here we are using the separable convolution layer to reduce the number of multiplications performed hence reducing computational cost
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # Batch normalization layer so that values stays between 0-255
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Third convolution block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Fourth convolution block
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        # Drop out layer is used to drop some data for better learning of the rest of the data 
        x = Dropout(rate=0.2)(x)

        # Fifth conv block
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # Fully connected layer
        # Flatten layer is to convert 2D data into 1D array
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=0.7)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(rate=0.3)(x)

        # Output layer
        output = Dense(units=1, activation='sigmoid')(x)


        keras.layers.Concatenate(axis=-1)
        # Creating model and compiling
        model = Model(inputs=inputs, outputs=output)

        # Callbacks
        checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)

        
        return model 
