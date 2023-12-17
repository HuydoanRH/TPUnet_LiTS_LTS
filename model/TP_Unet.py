import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .layer_utils import *


DEPTH = 4

def TP_Unet(input_shape,num_classes=1,dropout_rate=0.0,batch_norm=True,activation='relu',name='TP_Unet'):
    features = 64
    skip_connections = []
    pyramid_down = []
    futures_pyramid_down = []
    pyramid_up = []
    futures_pyramid_up = []

    inputs = layers.Input(input_shape)
    x = inputs

    # Downsampling layers
    for i in range(DEPTH):
        x = ConvBlock(features,(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)
        skip_connections.append(x)
        x = layers.MaxPool2D((2,2))(x)
        features = features*2

        x = ConvBlock(features,(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)
        skip_connections.append(x)
        pyramid_down.append(x)
        futures_pyramid_down.append(features)
        x = layers.MaxPool2D((2,2))(x)
        features = features*2

    # Bottom layers
    x = ConvBlock(features,(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)
    skip_connections.append(x)

    # Upsampling layers
    for i in reversed(range(DEPTH)):
        features = features/2
        if i != 0:
            att = TritentionBlock(features,name='tritention_'+str(i))(skip_connections[i],skip_connections[i+1],skip_connections[i-1])
        else:
            att = AttentionBlock(features,name='attention_'+str(i))(skip_connections[i],skip_connections[i+1])
        
        x = layers.Conv2DTranspose(features,(2,2),strides=(2,2),padding='same')(x)
        x = layers.Activation(activation)(x)
        x = layers.concatenate([x, att], axis=-1)
        x = ConvBlock(features,(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)
        pyramid_up.append(x)
        futures_pyramid_up.append(features)

    # Handle pyramid in midle of process.
    xm = layers.Conv2D(futures_pyramid_down[len(futures_pyramid_down) - 1], kernel_size=(1,1),padding='same')(pyramid_down[len(futures_pyramid_down) - 1])
    for index in reversed(range(len(futures_pyramid_down) - 1)):
        _xm = layers.Conv2D(futures_pyramid_down[index], kernel_size=(1,1),padding='same')(pyramid_down[index])
        xm = layers.Conv2DTranspose(futures_pyramid_down[index],(2,2),strides=(2,2),padding='same')(xm)
        xm = layers.Add()([xm, _xm])
        
    # Handle pyramid in end of process.
    xe = layers.Conv2D(futures_pyramid_up[0], kernel_size=(1,1),padding='same')(pyramid_up[0])
    for index in range(1,len(pyramid_up)):
        _xe = layers.Conv2D(futures_pyramid_up[index], kernel_size=(1,1),padding='same')(pyramid_up[index])
        xe = layers.Conv2DTranspose(futures_pyramid_up[index],(2,2),strides=(2,2),padding='same')(xe)
        xe = layers.Add()([xe, _xe])
    
    xe = ConvBlock(futures_pyramid_down[0],(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(xe)
    xm = ConvBlock(futures_pyramid_down[0],(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(xm)
    x =  ConvBlock(futures_pyramid_down[0],(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)

    conv_final = layers.concatenate([xe, xm], axis=-1)
    conv_final = layers.concatenate([conv_final, x], axis=-1)
    
    # Model head
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1),padding='same')(x)
    if batch_norm:
        conv_final = layers.BatchNormalization(axis=-1)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)
    model = keras.models.Model(inputs, conv_final, name=name)

    return model