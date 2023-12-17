import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .layer_utils import *


DEPTH = 4

def AttentionUNet(input_shape,num_classes=1,dropout_rate=0.0,batch_norm=True,activation='relu',name='att_unet'):
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

    # Upsampling layers
    for i in reversed(range(DEPTH)):
        features = features/2
        att = AttentionBlock(features,name='attention_'+str(i))(skip_connections[i],x)
        
        x = layers.Conv2DTranspose(features,(3,3),strides=(2,2),padding='same')(x)
        # x = layers.UpSampling2D(size=(2,2),data_format='chanel_last')
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