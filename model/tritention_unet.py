import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .layer_utils import *


DEPTH = 4

def TritentionUNet(input_shape,num_classes=1,dropout_rate=0.0,batch_norm=True,activation='relu',name='tri_unet'):
    features = 64
    skip_connections = []

    inputs = layers.Input(input_shape)
    x = inputs

    # Downsampling layers
    for i in range(DEPTH):
        x = ConvBlock(features,(3,3),padding='same',activation=activation,dropout=dropout_rate,use_batchnorm=batch_norm)(x)
        skip_connections.append(x)
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
    
    # Model head
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1),padding='same')(x)
    if batch_norm:
        conv_final = layers.BatchNormalization(axis=-1)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)
    model = keras.models.Model(inputs, conv_final, name=name)

    return model