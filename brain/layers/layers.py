#!/usr/bin/env python3
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

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1e-4,
        mult_factor: int = 1,
        channel_out_mult_factor: int = 4,
        downsampling: str = "conv",
        conditioning_layer: keras.layers.Layer = None,
        **kwargs,
    ):
        """
        ResNet-like encoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                            |___________________> Conv > BN >_________________|

        Args:
            filter_num: base number of used filters.
            dropout_rate: used dropout rate. A 0 value means no dropout.
            stride: applied downsampling stride.
            activation: a registered tf2 activation function.
            bn: whether use batch normalization (BN), Group Normalization (GN), or None
            groups: the number of groups for Group Normalization.
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            mult_factor: middle filter multiplicative factor
            channel_out_mult_factor: multiplicative factor for output channels
            downsampling: downsampling type; conv or pool
            conditioning_layer: a FiLM conditioning layer to apply to the second conv layer
            **kwargs: kwargs to pass to parent class
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
        self.conditioning_layer = conditioning_layer
        self.film2 = None

        return

    def build(self, input_shape):
        """Build layers."""
        self.conv1 = keras.layers.Conv3D(
            filters=self.filter_num,
            kernel_size=(1, 1, 1),
            strides=self.stride,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.conv2 = keras.layers.Conv3D(
            filters=self.filter_num * self.mult_factor,
            kernel_size=(3, 3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.conv3 = keras.layers.Conv3D(
            filters=self.filter_num * self.channel_out_mult_factor,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        # Downsampling
        if self.downsampling == "conv":
            self.downsample = keras.layers.Conv3D(
                filters=self.filter_num * self.channel_out_mult_factor,
                kernel_size=(1, 1, 1),
                strides=self.stride,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )
        elif self.downsampling == "pooling":
            self.downsample = keras.layers.MaxPool3D(pool_size=self.stride)
        else:
            raise ValueError(f"Downsampling {self.downsampling} not recognised!")

        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()
            self.bn_res = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        if self.conditioning_layer is not None:
            self.film2 = FiLM3DLayer(
                condition_dim = self.conditioning_layer.condition_dim
            )
        return

    def call(self, inputs, condition_vector=None, training=None):
        """Forward pass."""
        residual = self.downsample(inputs)
        if self.bn:
            residual = self.bn_res(residual, training=training)

        x = self.conv1(inputs)
        if self.bn:
            x = self.bn1(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv2(x)
        if self.conditioning_layer is not None and condition_vector is not None:
            x = self.film2([x, condition_vector])
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
            "downsampling": self.downsampling,
            "conditioning_layer": self.conditioning_layer.name if self.conditioning_layer else None
        }


class UpBottleNeck(keras.layers.Layer):
    """
    ResNet-like decoder bottleneck block for 3D tensors.

    Structure: - Input -|> Conv > BN > ConvT > BN > Conv > BN > Dropout > Add -
                        |___________________> ConvT > BN >_________________|
    """

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1.0e-4,
        mult_factor: int = 1,
        channel_out_mult_factor: int = 4,
        conditioning_layer: keras.layers.Layer = None,
        **kwargs,
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
            conditioning_layer: a FiLM conditioning layer to apply to the second conv layer
            **kwargs: other kwargs
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
        self.conditioning_layer = conditioning_layer

    def build(self, input_shape):
        """Build layers."""
        self.conv1 = keras.layers.Conv3D(
            filters=self.filter_num,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.conv2_up = keras.layers.Conv3DTranspose(
            filters=self.filter_num * self.mult_factor,
            kernel_size=(3, 3, 3),
            strides=self.stride,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.conv3 = keras.layers.Conv3D(
            filters=self.filter_num * self.channel_out_mult_factor,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.upsample = keras.Sequential()
        self.upsample.add(
            keras.layers.Conv3DTranspose(
                filters=self.filter_num * self.channel_out_mult_factor,
                kernel_size=(1, 1, 1),
                strides=self.stride,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )
        )
        if self.bn:
            self.upsample.add(keras.layers.BatchNormalization())

        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        if self.conditioning_layer is not None:
            self.film2 = FiLM3DLayer(
                condition_dim = self.conditioning_layer.condition_dim
            )
        return            

    def call(self, inputs, condition_vector=None, training=None, **kwargs):
        """Forward pass."""
        residual = self.upsample(inputs)

        x = self.conv1(inputs)
        if self.bn:
            x = self.bn1(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv2_up(x)
        if self.conditioning_layer is not None and condition_vector is not None:
            x = self.film2([x, condition_vector])
            
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
            "kernel_regularizer": self.kernel_regularizer_value,
            "conditioning_layer": self.conditioning_layer.name if self.conditioning_layer else None
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
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build weights using layer input shape."""

        self.num_features = input_shape[-1]
        self.scale = self.add_weight(
            name=self.name + "_scale",
            shape=(self.num_features,),
            initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.02),
            trainable=True,
        )
        self.shift = self.add_weight(
            name=self.name + "_shift",
            shape=(self.num_features,),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
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

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1.0e-4,
        n_conv_row: int = 1,
        downsampling: str = "conv",
        channel_out_mult_factor: int = 2,
        **kwargs,
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
        super().__init__(**kwargs)
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

        for _i in range(self.n_conv_row):
            self.convs.add(
                keras.layers.Conv3D(
                    filters=self.filter_num,  # Conv
                    kernel_size=(3, 3, 3),
                    strides=1,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )
            if self.bn:  # Batch norm
                self.convs.add(keras.layers.BatchNormalization())
            self.convs.add(keras.layers.Activation(self.activation))  # Activation

        # Downsampling
        if self.downsampling == "conv":
            self.downsample = keras.layers.Conv3D(
                filters=self.filter_num * self.channel_out_mult_factor,
                kernel_size=(1, 1, 1),
                strides=self.stride,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )
        elif self.downsampling == "pooling":
            self.downsample = keras.layers.MaxPool3D(pool_size=self.stride)
        else:
            raise ValueError(f"Downsampling {self.downsampling} not recognised!")

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
            "channel_out_mult_factor": self.channel_out_mult_factor,
        }


class Residual(keras.layers.Layer):
    """
    Residual encoder block for 3D tensors.
    Structure: - Input -|> {[Conv > BN > Activation] x 'n_conv_row' times (no last) > Add } x 'm_res_blocks' times > DS > Dropout > Add 
                        |_________________________________________________________________/
    """  # noqa: E501

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1.0e-4,
        n_conv_row: int = 1,
        m_res_blocks: int = 1,
        downsampling: str = "conv",
        channel_out_mult_factor: int = 2,
        **kwargs,
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
        """  # noqa: E501
        super().__init__(**kwargs)
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
            self.convs.add(
                keras.layers.Conv3D(
                    filters=input_shape[-1],  # Conv
                    kernel_size=(3, 3, 3),
                    strides=1,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )
            if self.bn:  # Batch norm
                self.convs.add(keras.layers.BatchNormalization())

            if i != (self.n_conv_row - 1):
                self.convs.add(keras.layers.Activation(self.activation))  # Activation

        # Downsampling
        if self.downsampling == "conv":
            self.downsample = keras.layers.Conv3D(
                filters=self.filter_num * self.channel_out_mult_factor,
                kernel_size=(1, 1, 1),
                strides=self.stride,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )
        elif self.downsampling == "pooling":
            self.downsample = keras.layers.MaxPool3D(pool_size=self.stride)
        else:
            raise ValueError(f"Downsampling {self.downsampling} not recognised!")

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
            "channel_out_mult_factor": self.channel_out_mult_factor,
        }


class UpPlain(keras.layers.Layer):
    """
    VGG-like encoder block for 3D tensors.
    Structure: - Input -> [Conv > BN > Activation] x 'n_conv_row' times > DS > Dropout > Activation
    """

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1.0e-4,
        n_conv_row=1,
        mult_factor: int = 1,
        channel_out_mult_factor: int = 4,
        **kwargs,
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
            groups: the number of groups for Group Normalization.
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            n_conv_row: number of convolutional repeats
            mult_factor: middle filter multiplicative factor
            channel_out_mult_factor: multiplicative factor for output channels
            **kwargs: kwargs for parent class
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row
        self.mult_factor = mult_factor
        self.channel_out_mult_factor = channel_out_mult_factor

    def build(self, input_shape):
        """Build layers."""
        # Define conv layers with Convs, BN, and activation
        self.convs = keras.Sequential()
        for _i in range(self.n_conv_row):
            self.convs.add(
                keras.layers.Conv3D(
                    filters=self.filter_num * self.mult_factor,  # Conv
                    kernel_size=(3, 3, 3),
                    strides=1,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )

            if self.bn:  # Batch norm
                self.convs.add(keras.layers.BatchNormalization())

            self.convs.add(keras.layers.Activation(self.activation))  # Activation

        self.up_conv = keras.layers.Conv3DTranspose(
            filters=self.filter_num * self.channel_out_mult_factor,
            kernel_size=[self.stride] * 3,
            strides=self.stride,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        """Forward pass."""
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)
        x = self.up_conv(x)
        x = getattr(keras.ops, self.activation)(x)
        return x

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
            "kernel_regularizer": self.kernel_regularizer_value,
        }


class ExpandNeck(keras.layers.Layer):
    """
    ResNet-like encoder "expand" neck block for 3D tensors.

    Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                    |___________________> Conv > BN >________________/
    """

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1.0e-4,
        mult_factor: int = 1,
        downsampling: str = "conv",
        channel_out_mult_factor: int = 1,
        **kwargs,
    ):
        """
        ResNet-like encoder "expand" block for 3D tensors.
        Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                            |___________________> Conv > BN >________________/

        Args:
            filter_num: base number of used filters.
            dropout_rate: used dropout rate. A 0 value means no dropout.
            stride: applied downsampling stride.
            activation: a registered tf2 activation function.
            bn: whether use batch normalization (BN), Group Normalization (GN), or None
            kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
            kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
            mult_factor: middle filter multiplicative factor
            downsampling: downsampling type (conv or pool)
            channel_out_mult_factor: multiplicative factor for output channels
            kwargs: kwargs for parent class
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor
        self.downsampling = downsampling
        self.channel_out_mult_factor = channel_out_mult_factor

        self.kernel_regularizer_value = kernel_regularizer
        self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """Build layers."""
        self.conv1 = keras.layers.Conv3D(
            filters=self.filter_num,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.conv2 = keras.layers.Conv3D(
            filters=self.filter_num * self.mult_factor,
            kernel_size=(3, 3, 3),
            strides=self.stride,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.conv3 = keras.layers.Conv3D(
            filters=self.filter_num * self.channel_out_mult_factor,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        if self.bn:
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()
            self.bn3 = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        # Downsampling
        self.downsample = keras.Sequential()
        if self.downsampling == "conv":
            self.downsample.add(
                keras.layers.Conv3D(
                    filters=self.filter_num * self.channel_out_mult_factor,
                    kernel_size=(1, 1, 1),
                    strides=self.stride,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )
        elif self.downsampling == "pooling":
            self.downsample.add(keras.layers.MaxPool3D(pool_size=self.stride))
        else:
            raise ValueError(f"Downsampling {self.downsampling} not recognised!")

        if self.bn:
            self.downsample.add(keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        """Forward pass."""
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ["BN", "GN"]:
            x = self.bn1(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv2(x)
        if self.bn in ["BN", "GN"]:
            x = self.bn2(x, training=training)
        x = getattr(keras.ops, self.activation)(x)

        x = self.conv3(x)
        if self.bn in ["BN", "GN"]:
            x = self.bn3(x, training=training)

        if training:
            x = self.dropout(x)
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
            "kernel_regularizer": self.kernel_regularizer_value,
        }

class FiLMConditioningVector(keras.layers.Layer):
    """ Keras layer to generate a conditioning vector for FiLM based on task ID. 
    
    This layer uses an embedding layer to create a conditioning vector of shape (batch_size, condition_dim).
    It is useful for tasks like segmentation where different tasks may require different conditioning vectors.
    Ex.
        from src.model.layers import FiLMConditioningVector
        cond_layer = FiLMConditioningVector(num_tasks=3, condition_dim=256)
        task_ids = torch.tensor([0, 1, 2])  # Example task IDs
        cond_vector = cond_layer(task_ids)
        print(cond_vector.shape)  # Output: (3, 256)
    """

    def __init__(self, 
                 num_tasks,
                 condition_dim=256,
                 l2_weight=1e-4,
                 **kwargs):
        """ Initialize the FiLMConditioningVector layer.

        Args:
            num_tasks (int): Number of task types (e.g., 2 for dual segmentation tasks).
            condition_dim (int): Dimension of the output conditioning vector.
            l2_weight (float): L2 regularization strength.
            kwargs: Additional keyword arguments for the layer.
        """
        super().__init__(**kwargs)

        self.num_tasks = num_tasks
        self.condition_dim = condition_dim
        self.l2 = keras.regularizers.L2(l2_weight)

        # Learnable embedding for each task
        self.task_embedding = keras.layers.Embedding(
            input_dim=num_tasks,
            output_dim=condition_dim,
            embeddings_regularizer=self.l2
        )

    def call(self, task_id) -> keras.layers.Layer:
        """
        Args:
            task_id: Integer tensor of shape (batch_size,) representing task indices.
        
        Returns:
            Tensor of shape (batch_size, condition_dim)
        """
        return self.task_embedding(task_id)

    def get_config(self) -> dict:
        """ Get the configuration of the layer for serialization. """
        return {
            "num_tasks": self.num_tasks,
            "condition_dim": self.condition_dim,
            "l2_weight": self.l2.l2
        }

    def compute_output_shape(self, input_shape):
        """ Compute the output shape of the layer based on input shape. """
        return (input_shape[0], self.condition_dim)
        

class FiLM3DLayer(keras.layers.Layer):
    """ FiLM layer for 3D feature modulation with L2 regularization. 
    This layer applies FiLM modulation to 3D tensors using learned parameters for gamma and beta.
    To insert this layer in a model, use:
    Ex. 
        conditioning_vector_layer = FiLMConditioningVector(num_tasks=2, condition_dim=256)
        film_layer = FiLM3DLayer(condition_dim=256)
        input_tensor = keras.Input(shape=(16, 16, 16, 64))  # Example input tensor
        task_ids = torch.tensor([0, 1, 1, 0])  # Example task IDs

        condition_vector = conditioning_vector_layer(task_ids)  # Get conditioning vector
        modulated_tensor = film_layer([input_tensor, condition_vector])
        print("Input shape:     ", input_tensor.shape)      # Output: (None, 16, 16, 16, 64)
        print("Condition shape: ", condition_vector.shape)  # Output: (4, 256)
        print("Output shape:    ", modulated_tensor.shape)  # Output: (None, 16, 16, 16, 64)
    """

    def __init__(self, condition_dim, l2_weight=1e-4, **kwargs):
        """
        FiLM layer for 3D feature modulation with L2 regularization.

        Args:
            condition_dim (int): Dimensionality of the conditioning vector.
            l2_weight (float): L2 regularization strength (weight decay).
            kwargs: Additional keyword arguments for the layer.
        """
        super().__init__(**kwargs)
        self.condition_dim = condition_dim
        self.gamma_dense = None
        self.beta_dense = None
        self.l2 = keras.regularizers.L2(l2_weight)

    def build(self, input_shape):
        """ Build the FiLM layer weights. """
        feature_channels = input_shape[0][-1]
        self.gamma_dense = keras.layers.Dense(
            feature_channels,
            kernel_regularizer=self.l2,
            bias_regularizer=self.l2,
            kernel_initializer='zeros',
            bias_initializer=keras.initializers.Constant(1.0)
        )
        self.beta_dense = keras.layers.Dense(
            feature_channels,
            kernel_regularizer=self.l2,
            bias_regularizer=self.l2,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )

    def call(self, inputs):
        """
        Apply FiLM modulation.

        Args:
            x: Tensor of shape (B, D, H, W, C)
            condition_vector: Tensor of shape (B, condition_dim)

        Returns:
            Modulated tensor of shape (B, D, H, W, C)
        """
        x, condition_vector = inputs
        gamma = self.gamma_dense(condition_vector)  # (B, C)
        beta = self.beta_dense(condition_vector)    # (B, C)

        channel_dim = x.shape[-1]
        gamma = keras.layers.Reshape(target_shape=(1, 1, 1, channel_dim))(gamma)
        beta = keras.layers.Reshape(target_shape=(1, 1, 1, channel_dim))(beta)

        return gamma * x + beta

    def compute_output_shape(self, input_shape):
        """ Compute the output shape of the FiLM layer based on the input shape. """
        return input_shape[0]
