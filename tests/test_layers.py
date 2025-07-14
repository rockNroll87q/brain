"""
Created on Friday - July 11 2025

@author: Austin Dibble, University of Glasgow
"""

import unittest
import os

os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy as np

import layers

# ===============================================
# *                BottleNeck                   *
# ===============================================
class TestBottleNeckLayer(unittest.TestCase):
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
        self.assertEqual(expected_shape, bottleneck.compute_output_shape(self.input_tensor.shape))

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
        self.assertEqual(expected_shape, bottleneck.compute_output_shape(self.input_tensor.shape))


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

    def testBottleNeckModelWithUnknownShapes_build_compile_train(self):
        """Test building, compiling, and one-step training of a model with unknown input shapes"""

        # Create Input layer with dynamic spatial dimensions
        input_layer = keras.Input(shape=(None, None, None, 16))  # Depth, Height, Width, Channels

        # Add the BottleNeck layer
        x = layers.BottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )(input_layer)

        # Add a global pooling and dense for simplicity
        x = keras.layers.GlobalAveragePooling3D()(x)
        output = keras.layers.Dense(1)(x)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)

        # Compile model
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy input and target data (must match input shape)
        dummy_input = keras.random.normal((1, 32, 32, 32, 16))
        dummy_target = keras.random.normal((1, 1))

        # Fit for one step
        model.fit(dummy_input, dummy_target, epochs=1, steps_per_epoch=1, verbose=0)

# ===============================================
# *                UpBottleNeck                 *
# ===============================================
class TestUpBottleNeckLayer(unittest.TestCase):
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

    def testUpBottleNeck_output_shape(self):
        """Test output shape and type of the BottleNeck layer"""
        bottleneck = layers.UpBottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )
        output = bottleneck(self.input_tensor)

        expected_shape = (
            self.input_shape[0],
            self.input_shape[1] * self.stride,
            self.input_shape[2] * self.stride,
            self.input_shape[3] * self.stride,
            self.filter_num * bottleneck.channel_out_mult_factor
        )

        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(expected_shape, bottleneck.compute_output_shape(self.input_tensor.shape))

        # Try modifying channel multiplier
        bottleneck = layers.UpBottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride,
            channel_out_mult_factor=1
        )
        output = bottleneck(self.input_tensor)

        expected_shape = (
            self.input_shape[0],
            self.input_shape[1] * self.stride,
            self.input_shape[2] * self.stride,
            self.input_shape[3] * self.stride,
            self.filter_num * bottleneck.channel_out_mult_factor
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(expected_shape, bottleneck.compute_output_shape(self.input_tensor.shape))

    def testUpBottleNeck_compilation(self):
        """Test if the layer integrates into a simple model and compiles"""
        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.UpBottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )(inputs)
        model = keras.Model(inputs, x)
        try:
            model.compile(optimizer='adam', loss='mse')
        except Exception as e:
            self.fail(f'Model compilation failed: {e}')

    def testUpBottleNeck_no_downsampling(self):
        """Test behavior when stride=1 (no spatial downsampling)"""
        bottleneck = layers.UpBottleNeck(
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

    def testUpBottleNeckModelWithUnknownShapes_build_compile_train(self):
        """Test building, compiling, and one-step training of a model with unknown input shapes"""

        # Create Input layer with dynamic spatial dimensions
        input_layer = keras.Input(shape=(None, None, None, 16))  # Depth, Height, Width, Channels

        # Add the BottleNeck layer
        x = layers.UpBottleNeck(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride
        )(input_layer)

        # Add a global pooling and dense for simplicity
        x = keras.layers.GlobalAveragePooling3D()(x)
        output = keras.layers.Dense(1)(x)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)

        # Compile model
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy input and target data (must match input shape)
        dummy_input = keras.random.normal((1, 32, 32, 32, 16))
        dummy_target = keras.random.normal((1, 1))

        # Fit for one step
        model.fit(dummy_input, dummy_target, epochs=1, steps_per_epoch=1, verbose=0)


class TestSSFAdaLayer(unittest.TestCase):
    """Test the SSFAdaLayer"""

    def test_channels_last_basic(self):
        """Test the layer with channels_last input (batch, H, W, C)."""
        layer = layers.SSFAdaLayer()
        input_tensor = keras.random.normal((2, 8, 8, 4))  # batch=2, H=8, W=8, channels=4
        output = layer(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(np.allclose(output.shape, input_tensor.shape))

    def test_channels_first_basic(self):
        """Test the layer with channels_first input (batch, C, H, W)."""
        layer = layers.SSFAdaLayer()
        input_tensor = keras.random.normal((2, 4, 8, 8))  # batch=2, channels=4, H=8, W=8
        output = layer(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(np.allclose(output.shape, input_tensor.shape))

    def test_in_model_with_undefined_shape(self):
        """Pass the layer an input with undefined batch shape via Conv2D."""
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 3)),
            keras.layers.Conv2D(4, kernel_size=3, padding='same'),
            layers.SSFAdaLayer()
        ])
        x = keras.random.normal((5, 16, 16, 3))  # batch=5
        y = model(x)
        self.assertEqual(y.shape, (5, 16, 16, 4))

    def test_fit_step_channels_last(self):
        """Compile model and run one training step with channels_last input."""
        inputs = keras.layers.Input(shape=(16, 16, 3))
        x = keras.layers.Conv2D(4, 3, padding='same')(inputs)
        x = layers.SSFAdaLayer()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')

        x_data = np.random.randn(4, 16, 16, 3).astype(np.float32)
        y_data = np.random.randn(4, 1).astype(np.float32)

        history = model.fit(x_data, y_data, epochs=1, batch_size=2, verbose=0)
        loss = history.history['loss'][0]
        self.assertTrue(np.isfinite(loss))

    def test_fit_step_channels_first(self):
        """Compile model and run one training step with channels_first input."""
        keras.backend.set_image_data_format('channels_first')

        inputs = keras.layers.Input(shape=(3, 16, 16))
        x = keras.layers.Conv2D(4, 3, padding='same', data_format='channels_first')(inputs)
        x = layers.SSFAdaLayer()(x)
        x = keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')

        x_data = np.random.randn(4, 3, 16, 16).astype(np.float32)
        y_data = np.random.randn(4, 1).astype(np.float32)

        history = model.fit(x_data, y_data, epochs=1, batch_size=2, verbose=0)
        loss = history.history['loss'][0]
        self.assertTrue(np.isfinite(loss))

        # Reset to default to avoid side effects
        keras.backend.set_image_data_format('channels_last')

    def test_invalid_shape_raises(self):
        """Test that mismatched input raises ValueError."""
        layer = layers.SSFAdaLayer()
        # Build with channels=4
        layer.build((None, 8, 8, 4))
        # Now pass input with incompatible shape
        bad_input = keras.random.normal((2, 8, 8, 5))
        with self.assertRaises(ValueError):
            _ = layer(bad_input)


if __name__ == "__main__":
    unittest.main()
