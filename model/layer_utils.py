import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def ConvBlock(filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               activation=None,
               kernel_initializer='he_uniform',
               dropout=None,
               use_batchnorm=True):
    def wrapper(input_tensor):
        conv = layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=(1, 1),
                             padding=padding,
                             kernel_initializer=kernel_initializer
                            )(input_tensor)
        if use_batchnorm:
            conv = layers.BatchNormalization()(conv)
        if activation:
            conv = layers.Activation(activation)(conv)

        conv = layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=(1, 1),
                             padding=padding,
                             kernel_initializer=kernel_initializer
                            )(conv)
        if use_batchnorm:
            conv = layers.BatchNormalization()(conv)
        if activation:
            conv = layers.Activation(activation)(conv)
        
        if 0 < dropout <= 1:
            conv = layers.Dropout(dropout)(conv)
        return conv

    return wrapper


def AttentionBlock(inter_filters,name=None):
    def wrapper(x,gating):
        shape_x = K.int_shape(x)        # (1, 128, 128, 128)
        shape_g = K.int_shape(gating)   # (1, 64, 64, 64)

        theta_x = layers.Conv2D(inter_filters,(2,2),strides=(2,2),padding='same')(x)    # (1, 64, 64, 128)
        phi_g   = layers.Conv2D(inter_filters,(1,1),padding='same')(gating) # (1, 64, 64, 128)

        concate_xg  = layers.add([theta_x,phi_g])   # (1, 64, 64, 128)
        relu_xg     = layers.Activation('relu')(concate_xg) # (1, 64, 64, 128)
        psi         = layers.Conv2D(1,(1,1),padding='same')(relu_xg)    # (1, 64, 64, 1)
        sigmoid_xg  = layers.Activation('sigmoid')(psi) # (1, 64, 64, 1)
        up_psi      = layers.UpSampling2D((shape_x[1]//shape_g[1],shape_x[2]//shape_g[2]),
                                          name=f'{name}_coefficient' if name else None)(sigmoid_xg)  # (1, 128, 128, 1)
        up_psi      = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), 
                            arguments={'repnum': shape_x[3]})(up_psi)   # (1, 128, 128, 128)
        y = layers.multiply([up_psi, x])    # (1, 128, 128, 128)

        result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)   # (1, 128, 128, 128)
        result_bn = layers.BatchNormalization(name=name)(result) # (1, 128, 128, 128)

        return result_bn

    return wrapper


def TritentionBlock(inter_filters,name=None):
    def wrapper(x,lower_gate, upper_gate):
        shape_x = K.int_shape(x)            
        shape_lg = K.int_shape(lower_gate)
        shape_ug = K.int_shape(upper_gate)

        theta_x = layers.Conv2D(inter_filters, kernel_size=(1,1), strides=(1,1), padding='same')(x)
        phi_lg  = layers.Conv2DTranspose(inter_filters, kernel_size=(1,1), strides=(2,2), padding='same')(lower_gate)
        phi_ug  = layers.Conv2D(inter_filters, kernel_size=(2,2), strides=(2,2), padding='same')(upper_gate)

        concate_xlg = layers.add([theta_x, phi_lg])
        concate_xug = layers.add([theta_x, phi_ug])

        relu_xlg    = layers.Activation('relu')(concate_xlg)
        relu_xug    = layers.Activation('relu')(concate_xug)

        conv_xlg    = layers.Conv2D(inter_filters/2, kernel_size=(1,1), strides=(1,1), padding='same')(relu_xlg)
        conv_xug    = layers.Conv2D(inter_filters/2, kernel_size=(1,1), strides=(1,1), padding='same')(relu_xug)

        concate_xlu = layers.add([conv_xlg, conv_xug])
        relu_xlu    = layers.Activation('relu')(concate_xlu)
        conv_xlu    = layers.Conv2D(1, kernel_size=(1,1), strides=(1,1), padding='same')(relu_xlu)
        sigmoid_xlu = layers.Activation('sigmoid', name=f'{name}_coefficient' if name else None)(conv_xlu)
        up_xlu      = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), 
                            arguments={'repnum': shape_x[-1]})(sigmoid_xlu)
        y = layers.multiply([up_xlu, x])
        
        result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization(name=name)(result)

        return result_bn

    return wrapper