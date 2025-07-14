#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Monday - July 07 2025, 13:39:35

@author: Connor Dalby, University of Glasgow
@author: Austin Dibble, University of Glasgow
@author: Michele Svanera, University of Glasgow

"""
import keras

class BottleNeck(keras.layers.Layer):
    """
    ResNet-like encoder bottleneck block for 3D tensors.

    Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                        |___________________> Conv > BN >_________________|
    """
    def __init__(self, 
                 filter_num:int, 
                 dropout_rate:float,
                 stride:int = 2,
                 activation:str = 'relu',
                 bn:bool = True,
                 kernel_initializer:str = 'he_normal',
                 kernel_regularizer:float = 1e-4,
                 mult_factor:int = 1,
                 channel_out_mult_factor:int = 4,
                 **kwargs
                 ):
        """
        ResNet-like encoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                            |___________________> Conv > BN >_________________|
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter multiplicative factor
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor
        self.channel_out_mult_factor = channel_out_mult_factor
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)
        
        return
    
    def build(self, input_shape):
        """Build layers."""
        self.conv1 = keras.layers.Conv3D(filters=self.filter_num,
                                         kernel_size=(1,1,1),
                                         strides=self.stride,
                                         padding='same',
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer
                                         )
        self.conv2 = keras.layers.Conv3D(filters=self.filter_num * self.mult_factor,
                                         kernel_size=(3, 3, 3),
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer
                                         )
        self.conv3 = keras.layers.Conv3D(filters=self.filter_num * self.channel_out_mult_factor,
                                        kernel_size=(1, 1, 1),
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=self.kernel_regularizer
                                        )
        
        self.downsample = keras.layers.Conv3D(filters=self.filter_num * self.channel_out_mult_factor,
                                        kernel_size=(1, 1, 1),
                                        strides=self.stride,
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=self.kernel_regularizer
                                        )
        
        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()
            self.bn_res = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        return
    
    def call(self, inputs, training=None):
        """Forward pass."""
        residual = self.downsample(inputs)
        if self.bn:
            residual = self.bn_res(residual, training=training)

        x = self.conv1(inputs)
        if self.bn:
            x = self.bn1(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x, training=training)
        
        x = self.dropout(x, training=training)

        output = getattr(keras.ops, self.activation)(keras.layers.add([residual, x]))

        return output
    
    def get_config(self):
        """Serialize our config options."""

        return {
            "filter_num": self.filter_num,
            "dropout_rate": self.dropout_rate,
            "stride": self.stride,
            "activation": self.activation,
            "bn": self.bn,
            "mult_factor": self.mult_factor,
            "channel_out_mult_factor": self.channel_out_mult_factor,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer_value
        }


class UpBottleNeck(keras.layers.Layer):
    """
    ResNet-like decoder bottleneck block for 3D tensors.

    Structure: - Input -|> Conv > BN > ConvT > BN > Conv > BN > Dropout > Add -
                        |___________________> ConvT > BN >_________________|
    """

    def __init__(self, 
                 filter_num:int,
                 dropout_rate:float,
                 stride:int = 2,
                 activation:str = 'relu', 
                 bn: bool = True,
                 kernel_initializer:str='he_normal',
                 kernel_regularizer:float=1.e-4,
                 mult_factor:int = 1,
                 channel_out_mult_factor:int = 4,
                 **kwargs
                 ):
        """
        ResNet-like decoder "up" bottleneck block for 3D tensors.
        Unlike the normal BottleNeck block, this one increases the output volume size

        Structure: - Input -|> Conv > BN > ConvT > BN > Conv > BN > Dropout > Add -
                            |___________________> ConvT > BN >_________________|
        
        Args:
            filter_num: base number of used filters.
            dropout_rate: used dropout rate. A 0 value means no dropout.
            stride: applied upsampling stride.
            activation: a registered tf2 activation function.
            bn: whether use batch normalization (BN), Group Normalization (GN), or None        
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            mult_factor: middle filter multiplicative factor
            channel_out_mult_factor: output filter multiplicative factor
            **kwargs: other kwargs
        """
        super(UpBottleNeck, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor
        self.channel_out_mult_factor = channel_out_mult_factor
        self.kernel_regularizer_value = kernel_regularizer
        kernel_regularizer = keras.regularizers.l2(kernel_regularizer)

        self.conv1 = keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2_up = keras.layers.Conv3DTranspose(filters=filter_num * self.mult_factor,
                                                        kernel_size=(3, 3, 3),
                                                        strides=stride,
                                                        padding='same',
                                                        kernel_initializer=kernel_initializer,
                                                        kernel_regularizer=kernel_regularizer
                                                        )
        self.conv3 = keras.layers.Conv3D(filters=filter_num * self.channel_out_mult_factor,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer)

        self.upsample = keras.Sequential()
        self.upsample.add(
            keras.layers.Conv3DTranspose(filters=filter_num * self.channel_out_mult_factor,
                                         kernel_size=(1, 1, 1),
                                         strides=stride,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer
                                        )
                        )
        if self.bn:
            self.upsample.add(keras.layers.BatchNormalization())

        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        """Forward pass."""
        residual = self.upsample(inputs)

        x = self.conv1(inputs)
        if self.bn:
            x = self.bn1(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv2_up(x)
        if self.bn:
            x = self.bn2(x, training=training)
        x = getattr(keras.ops, self.activation)(x)
        
        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x, training=training)

        if training:
            x = self.dropout(x)

        output = getattr(keras.ops, self.activation)(keras.layers.add([residual, x]))

        return output

    # def compute_output_shape(self, input_shape):
    #     batch_size, d, h, w, c = input_shape

    #     d_out = d * self.stride if d is not None else None
    #     h_out = h * self.stride if h is not None else None
    #     w_out = w * self.stride if w is not None else None

    #     c_out = self.filter_num * 4

    #     return (batch_size, d_out, h_out, w_out, c_out)
        
    def get_config(self):
        """Get serializable config"""
        return {
                "filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "mult_factor": self.mult_factor,
                "channel_out_mult_factor": self.channel_out_mult_factor
                }