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
                 downsampling:str='conv',
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
        self.downsampling = downsampling
        
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
                # Downsampling
        if self.downsampling == 'conv':
            self.downsample = keras.layers.Conv3D(filters = self.filter_num * self.channel_out_mult_factor,
                                                    kernel_size = (1, 1, 1),
                                                    strides = self.stride,
                                                    kernel_initializer = self.kernel_initializer,
                                                    kernel_regularizer = self.kernel_regularizer,
                                                    )
        elif self.downsampling == 'pooling':
            self.downsample = keras.layers.MaxPool3D(pool_size = self.stride)
        else:
            raise ValueError(f'Downsampling {self.downsampling} not recognised!')
        
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
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of this layer."""
        batch_size, d, h, w, c = input_shape

        d_out = d // self.stride if d is not None else None
        h_out = h // self.stride if h is not None else None
        w_out = w // self.stride if w is not None else None

        c_out = self.filter_num * self.channel_out_mult_factor

        return (batch_size, d_out, h_out, w_out, c_out)
    
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
            "kernel_regularizer": self.kernel_regularizer_value,
            "downsampling": self.downsampling
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
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)

    def build(self, input_shape):
        """Build layers."""
        self.conv1 = keras.layers.Conv3D(filters=self.filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer
                                            )
        self.conv2_up = keras.layers.Conv3DTranspose(filters=self.filter_num * self.mult_factor,
                                                        kernel_size=(3, 3, 3),
                                                        strides=self.stride,
                                                        padding='same',
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer
                                                        )
        self.conv3 = keras.layers.Conv3D(filters=self.filter_num * self.channel_out_mult_factor,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer)

        self.upsample = keras.Sequential()
        self.upsample.add(
            keras.layers.Conv3DTranspose(filters=self.filter_num * self.channel_out_mult_factor,
                                         kernel_size=(1, 1, 1),
                                         strides=self.stride,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer
                                        )
                        )
        if self.bn:
            self.upsample.add(keras.layers.BatchNormalization())

        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

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

    def compute_output_shape(self, input_shape):
        """Compute the output shape of this layer."""
        batch_size, d, h, w, c = input_shape

        d_out = d * self.stride if d is not None else None
        h_out = h * self.stride if h is not None else None
        w_out = w * self.stride if w is not None else None

        c_out = self.filter_num * self.channel_out_mult_factor

        return (batch_size, d_out, h_out, w_out, c_out)
        
    def get_config(self):
        """Get serializable config"""
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
    
class SSFAdaLayer(keras.layers.Layer):
    """
    Scale-Shift Feature Layers from https://arxiv.org/abs/2210.08823
    Intended for high-efficiency fine-tuning as a very simple form of FiLM layer.
    Intended to be added (during fine-tuning) between frozen model layers to 
    rescale the layer channels.
    """

    def __init__(self, **kwargs):
        """
        Scale-Shift Feature Layers.
        
        Args:
            **kwargs: Any kwargs to pass to parent
        """
        super(SSFAdaLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build weights using layer input shape."""

        self.num_features = input_shape[-1]
        self.scale = self.add_weight(
            name=self.name + '_scale',
            shape=(self.num_features,),
            initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.02),
            trainable=True
        )
        self.shift = self.add_weight(
            name=self.name + '_shift',
            shape=(self.num_features,),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )

        self.reshape_target = [1, self.num_features] + [1] * (len(input_shape) - 2)
        self.reshape_scale = keras.layers.Reshape(self.reshape_target)
        self.reshape_shift = keras.layers.Reshape(self.reshape_target)

    def call(self, inputs):
        """
        Apply learned scale and shift to the input tensor.

        Behavior depends on the layout of the input:
        - If the last dimension matches the number of features (channels_last),
        applies elementwise scale and shift directly.
        - If the second dimension matches the number of features (channels_first),
        reshapes the scale and shift parameters to broadcast over spatial dims.

        Args:
            inputs: Input tensor of shape (..., num_features) or (batch, num_features, ...)

        Returns:
            Tensor with the same shape as `inputs`, scaled and shifted.

        Raises:
            ValueError: If input shape is incompatible with the scale/shift dimensions.
        """
        if inputs.shape[-1] == self.num_features:
            # channels_last format or dense output
            return inputs * self.scale + self.shift
        elif inputs.shape[1] == self.num_features:
            # channels_first format
            scale = self.reshape_scale(self.scale)
            shift = self.reshape_shift(self.shift)
            return inputs * scale + shift
        else:
            raise ValueError("Input shape incompatible with scale/shift dimensions.")
        
    def compute_output_shape(self, input_shape):
        """Compute the output shape of this layer."""
        return input_shape
        
class Plain(keras.layers.Layer):
    """
    VGG-like encoder block for 3D tensors.
    Structure: - Input -> [Conv > BN > Activation] x 'n_conv_row' times > DS > Dropout > Activation
    """
    def __init__(self, 
                 filter_num:int, 
                 dropout_rate:float = 0.1,
                 stride:int = 2, 
                 activation:str = 'relu', 
                 bn:bool = True,
                 kernel_initializer:str = 'he_normal', 
                 kernel_regularizer:float = 1.e-4,
                 n_conv_row:int = 1, 
                 downsampling:str = 'conv', 
                 channel_out_mult_factor:int = 2,
                 **kwargs
                 ):
        """
        VGG-like encoder block for 3D tensors.
        Structure: - Input -> [Conv > BN > Activation] x 'n_conv_row' times > DS > Dropout > Activation
        
        Args:
            filter_num: base number of used filters.
            dropout_rate: used dropout rate. A 0 value means no dropout.
            stride: applied downsampling stride.
            activation: a registered tf2 activation function.
            bn: whether use batch normalization (BN), Group Normalization (GN), or None
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            n_conv_row: number of conv repetitions
            downsampling: downsampling strategy
            **kwargs: other kwargs passed to parent layer
            channel_out_mult_factor: the multiplier for the # of output channels
        """
        super(Plain, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row
        self.downsampling = downsampling
        self.channel_out_mult_factor = channel_out_mult_factor

    def build(self, input_shape):
        """Build layers."""
        # Define conv layers with Convs, BN, and activation
        self.convs = keras.Sequential()
        
        for i in range(self.n_conv_row):
            self.convs.add(keras.layers.Conv3D(filters = self.filter_num,  # Conv
                                                  kernel_size = (3, 3, 3),
                                                  strides = 1,
                                                  padding = 'same',
                                                  kernel_initializer = self.kernel_initializer,
                                                  kernel_regularizer = self.kernel_regularizer,
                                                  ))
            if self.bn:  # Batch norm
                self.convs.add(keras.layers.BatchNormalization())
            self.convs.add(keras.layers.Activation(self.activation))  # Activation

        # Downsampling
        if self.downsampling == 'conv':
            self.downsample = keras.layers.Conv3D(filters = self.filter_num * self.channel_out_mult_factor,
                                                    kernel_size = (1, 1, 1),
                                                    strides = self.stride,
                                                    padding = 'same',
                                                    kernel_initializer = self.kernel_initializer,
                                                    kernel_regularizer = self.kernel_regularizer,
                                                    )
        elif self.downsampling == 'pooling':
            self.downsample = keras.layers.MaxPool3D(pool_size = self.stride)
        else:
            raise ValueError(f'Downsampling {self.downsampling} not recognised!')

        # Dropout
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)
        x = self.downsample(x)
        x = getattr(keras.ops, self.activation)(x)
        return x

    def compute_output_shape(self, input_shape):
        """Compute the output shape of this layer."""
        batch_size, d, h, w, c = input_shape

        d_out = d // self.stride if d is not None else None
        h_out = h // self.stride if h is not None else None
        w_out = w // self.stride if w is not None else None

        c_out = self.filter_num * self.channel_out_mult_factor

        return (batch_size, d_out, h_out, w_out, c_out)

    def get_config(self):
        """Get serializable config"""
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "downsampling": self.downsampling,
                "kernel_regularizer": self.kernel_regularizer_value,
                "kernel_initializer": self.kernel_initializer,
                "channel_out_mult_factor": self.channel_out_mult_factor
                }

class Residual(keras.layers.Layer):
    """
    Residual encoder block for 3D tensors.
    Structure: - Input -|> {[Conv > BN > Activation] x 'n_conv_row' times (no last) > Add } x 'm_res_blocks' times > DS > Dropout > Add
                        |_________________________________________________________________/
    """
    def __init__(self, 
                 filter_num:int,
                 dropout_rate:float = 0.1,
                 stride:int = 2,
                 activation:str = 'relu',
                 bn:bool = True,
                 kernel_initializer:str = 'he_normal',
                 kernel_regularizer:float = 1.e-4,
                 n_conv_row:int = 1,
                 m_res_blocks:int = 1,
                 downsampling:str = 'conv', 
                 channel_out_mult_factor:int = 2,
                 **kwargs
                ):
        """
        Residual encoder block for 3D tensors.
        Structure: - Input -|> {[Conv > BN > Activation] x 'n_conv_row' times (no last) > Add } x 'm_res_blocks' times > DS > Dropout > Add
                            |_________________________________________________________________/

        Args:
            filter_num: base number of used filters.
            dropout_rate: used dropout rate. A 0 value means no dropout.
            stride: applied downsampling stride.
            activation: a registered tf2 activation function.
            bn: whether use batch normalization (BN), Group Normalization (GN), or None
            groups: the number of groups for Group Normalization.
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            n_conv_row: number of conv repetitions
            m_res_blocks: number of residual block repetitions
            downsampling: downsampling strategy
            channel_out_mult_factor: multiplicative factor for output channels
            **kwargs: kwargs for parent class
        """
        super(Residual, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row
        self.m_res_blocks = m_res_blocks
        self.channel_out_mult_factor = channel_out_mult_factor
        self.downsampling = downsampling

    def build(self, input_shape):
        """Build layers."""
        # Define conv layers with Convs, BN, and activation
        self.convs = keras.Sequential()
        for i in range(self.n_conv_row):
            self.convs.add(keras.layers.Conv3D(filters = input_shape[-1],  # Conv
                                                  kernel_size = (3, 3, 3),
                                                  strides = 1,
                                                  padding = 'same',
                                                  kernel_initializer = self.kernel_initializer,
                                                  kernel_regularizer = self.kernel_regularizer,
                                                  ))
            if self.bn:  # Batch norm
                self.convs.add(keras.layers.BatchNormalization())

            if i != (self.n_conv_row - 1):
                self.convs.add(keras.layers.Activation(self.activation))  # Activation

        # Downsampling
        if self.downsampling == 'conv':
            self.downsample = keras.layers.Conv3D(filters = self.filter_num * self.channel_out_mult_factor,
                                                    kernel_size = (1, 1, 1),
                                                    strides = self.stride,
                                                    kernel_initializer = self.kernel_initializer,
                                                    kernel_regularizer = self.kernel_regularizer,
                                                    )
        elif self.downsampling == 'pooling':
            self.downsample = keras.layers.MaxPool3D(pool_size = self.stride)
        else:
            raise ValueError(f'Downsampling {self.downsampling} not recognised!')

        # Dropout
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        """Forward pass."""
        x = self.convs(inputs)

        x = keras.layers.add([inputs, x])

        x = self.downsample(x)
        if training:
            x = self.dropout(x)

        x = getattr(keras.ops, self.activation)(x)

        return x
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of this layer."""
        batch_size, d, h, w, c = input_shape

        d_out = d // self.stride if d is not None else None
        h_out = h // self.stride if h is not None else None
        w_out = w // self.stride if w is not None else None

        c_out = self.filter_num * self.channel_out_mult_factor

        return (batch_size, d_out, h_out, w_out, c_out)

    def get_config(self):
        """Get serializable config"""
        return {
                "filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "downsampling": self.downsampling,
                "kernel_regularizer": self.kernel_regularizer_value,
                "kernel_initializer": self.kernel_initializer,
                "channel_out_mult_factor": self.channel_out_mult_factor
                }

# class UpPlain(tf.keras.layers.Layer):
#     def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
#                  groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, mult_factor: int = 1,
#                  **kwargs):
#         """
#         VGG-like encoder block for 3D tensors.

#         Structure: - Input -|> Conv > Conv Transpose > BN > Dropout -

#         :param filter_num: base number of used filters.
#         :param dropout_rate: used dropout rate. A 0 value means no dropout.
#         :param stride: applied downsampling stride.
#         :param activation: a registered tf2 activation function.
#         :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
#         :param groups: the number of groups for Group Normalization.
#         :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
#         :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
#         :param mult_factor: middle filter multiplicative factor
#         """
#         super(UpPlain, self).__init__(**kwargs)
#         self.activation = activation
#         self.filter_num = filter_num
#         self.dropout_rate = dropout_rate
#         self.stride = stride
#         self.bn = bn
#         kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
#         self.n_conv_row = n_conv_row
#         self.mult_factor = mult_factor

#         # Define conv layers with Convs, BN, and activation
#         self.convs = tf.keras.Sequential()
#         for i in range(self.n_conv_row):
#             self.convs.add(tf.keras.layers.Conv3D(filters=filter_num * self.mult_factor,  # Conv
#                                                   kernel_size=(3, 3, 3),
#                                                   strides=1,
#                                                   padding='same',
#                                                   kernel_initializer=kernel_initializer,
#                                                   kernel_regularizer=kernel_regularizer,
#                                                   ))
#             if self.bn == 'BN':  # Batch norm
#                 self.convs.add(tf.keras.layers.BatchNormalization())
#             elif self.bn == 'GN':
#                 self.convs.add(tf.keras.layers.GroupNormalization(groups=min(groups,filter_num)))
#             self.convs.add(tf.keras.layers.Activation(self.activation))  # Activation

#         self.up_conv = tf.keras.layers.Conv3DTranspose(filters=filter_num * 4,
#                                                        kernel_size=(stride, stride, stride),
#                                                        strides=stride,
#                                                        padding='same',
#                                                        kernel_initializer=kernel_initializer,
#                                                        kernel_regularizer=kernel_regularizer
#                                                        )

#         self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

#     def call(self, inputs, training=None, **kwargs):
#         x = self.convs(inputs, training=training)
#         if training:
#             x = self.dropout(x)
#         x = self.up_conv(x)
#         x = getattr(tf.nn, self.activation)(x)
#         return x

#     def get_config(self):
#         return {"filter_num": self.filter_num,
#                 "dropout_rate": self.dropout_rate,
#                 "stride": self.stride,
#                 "activation": self.activation,
#                 "bn": self.bn,
#                 "n_conv_row": self.n_conv_row,
#                 "mult_factor": self.mult_factor
#                 }

#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         return cls(**config)
    
# class ExpandNeck(tf.keras.layers.Layer):
#     def __init__(self, filter_num: int, dropout_rate: float = 0.1, stride: int = 2, activation: str = 'relu', bn: bool = True,
#                  groups: int = 8, kernel_initializer: str = 'he_normal', kernel_regularizer: float = 1.e-4,
#                  n_conv_row: int = 1, mult_factor: int = 4, downsampling: str = 'conv', **kwargs):
#         """
#         ResNet-like encoder bottleneck block for 3D tensors.
#         Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
#                             \___________________> Conv > BN >________________/
#         :param filter_num: base number of used filters.
#         :param dropout_rate: used dropout rate. A 0 value means no dropout.
#         :param stride: applied downsampling stride.
#         :param activation: a registered tf2 activation function.
#         :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
#         :param groups: the number of groups for Group Normalization.
#         :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
#         :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
#         :param mult_factor: middle filter multiplicative factor
#         """
#         super(ExpandNeck, self).__init__(**kwargs)
#         self.activation = activation
#         self.filter_num = filter_num
#         self.dropout_rate = dropout_rate
#         self.stride = stride
#         self.bn = bn
#         self.mult_factor = mult_factor
#         self.downsampling = downsampling

#         kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

#         self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
#                                             kernel_size=(1, 1, 1),
#                                             strides=1,
#                                             padding='same',
#                                             kernel_initializer=kernel_initializer,
#                                             kernel_regularizer=kernel_regularizer
#                                             )
#         self.conv2 = tf.keras.layers.Conv3D(filters=filter_num * self.mult_factor,
#                                             kernel_size=(3, 3, 3),
#                                             strides=stride,
#                                             padding='same',
#                                             kernel_initializer=kernel_initializer,
#                                             kernel_regularizer=kernel_regularizer
#                                             )
#         self.conv3 = tf.keras.layers.Conv3D(filters=filter_num,
#                                             kernel_size=(1, 1, 1),
#                                             strides=1,
#                                             padding='same',
#                                             kernel_initializer=kernel_initializer,
#                                             kernel_regularizer=kernel_regularizer
#                                             )

#         if self.bn == 'BN':
#             self.bn1 = tf.keras.layers.BatchNormalization()
#             self.bn2 = tf.keras.layers.BatchNormalization()
#             self.bn3 = tf.keras.layers.BatchNormalization()
#         elif self.bn == 'GN':
#             self.bn1 = tf.keras.layers.GroupNormalization(groups=min(groups,filter_num))
#             self.bn2 = tf.keras.layers.GroupNormalization(groups=min(groups,filter_num))
#             self.bn3 = tf.keras.layers.GroupNormalization(groups=min(groups,filter_num))

#         self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

#         # Downsampling
#         self.downsample = tf.keras.Sequential()
#         if downsampling == 'conv':
#             self.downsample.add(tf.keras.layers.Conv3D(filters = filter_num,
#                                                     kernel_size = (1, 1, 1),
#                                                     strides = self.stride,
#                                                     padding = 'same',
#                                                     kernel_initializer = kernel_initializer,
#                                                     kernel_regularizer = kernel_regularizer,
#                                                     ))
#         elif downsampling == 'pooling':
#             self.downsample.add(tf.keras.layers.MaxPool3D(pool_size = self.stride))
#         else:
#             logger.error(f'ERROR 19.0: config.network.downsampling {downsampling} not recognised!')
#             sys.exit(0)

#         if self.bn == 'BN':
#             self.downsample.add(tf.keras.layers.BatchNormalization())
#         elif self.bn == 'GN':
#             self.downsample.add(tf.keras.layers.GroupNormalization(groups=min(groups,filter_num)))


#     def call(self, inputs, training=None, **kwargs):
#         residual = self.downsample(inputs)

#         x = self.conv1(inputs)
#         if self.bn in ['BN', 'GN']:
#             x = self.bn1(x, training=training)
#         x = getattr(tf.nn, self.activation)(x)
#         x = self.conv2(x)
#         if self.bn in ['BN', 'GN']:
#             x = self.bn2(x, training=training)
#         x = getattr(tf.nn, self.activation)(x)
#         x = self.conv3(x)
#         if self.bn in ['BN', 'GN']:
#             x = self.bn3(x, training=training)
#         if training:
#             x = self.dropout(x)
#         output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))

#         return output

#     def get_config(self):
#         return {"filter_num": self.filter_num,
#                 "dropout_rate": self.dropout_rate,
#                 "stride": self.stride,
#                 "activation": self.activation,
#                 "bn": self.bn,
#                 "mult_factor": self.mult_factor
#                 }

#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         return cls(**config)