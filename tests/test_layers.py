"""
Created on Friday - July 11 2025

@author: Austin Dibble, University of Glasgow
"""

import unittest

import keras

import layers

class TestLayers(unittest.TestCase):
    """
    A unittest module for testing all code relating to the
    custom layers in the model.
    """

    def setUp(self):
        """Setup variables for the unittest module"""
        self.input_shape = (1, 32, 32, 32, 16)  # Batch, Depth, Height, Width, Channels
        self.filter_num = 32
        self.dropout_rate = 0.1
        self.stride = 2
        self.input_tensor = keras.random.normal(self.input_shape)

    # ===============================================
    # *                BottleNeck                   *
    # ===============================================
    def testBottleNeck_output_shape(self):
        """Test output shape and type of the BottleNeck layer"""
        bottleneck = layers.BottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )
        output = bottleneck(self.input_tensor)

        expected_shape = (
            self.input_shape[0],
            self.input_shape[1] // self.stride,
            self.input_shape[2] // self.stride,
            self.input_shape[3] // self.stride,
            self.filter_num * bottleneck.channel_out_mult_factor
        )

        self.assertEqual(output.shape, expected_shape)

        # Try modifying channel multiplier
        bottleneck = layers.BottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride,
            channel_out_mult_factor=1
        )
        output = bottleneck(self.input_tensor)

        expected_shape = (
            self.input_shape[0],
            self.input_shape[1] // self.stride,
            self.input_shape[2] // self.stride,
            self.input_shape[3] // self.stride,
            self.filter_num * bottleneck.channel_out_mult_factor
        )
        self.assertEqual(output.shape, expected_shape)

    def testBottleNeck_compilation(self):
        """Test if the layer integrates into a simple model and compiles"""
        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.BottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )(inputs)
        model = keras.Model(inputs, x)
        try:
            model.compile(optimizer='adam', loss='mse')
        except Exception as e:
            self.fail(f'Model compilation failed: {e}')

    def testBottleNeck_no_downsampling(self):
        """Test behavior when stride=1 (no spatial downsampling)"""
        bottleneck = layers.BottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=1
        )
        output = bottleneck(self.input_tensor)

        expected_shape = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
            self.filter_num * bottleneck.channel_out_mult_factor
        )
        self.assertEqual(output.shape, expected_shape)

    # ===============================================
    # *                BottleNeck                   *
    # ===============================================

if __name__ == "__main__":
    unittest.main()
