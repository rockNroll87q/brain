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


class BaseBottleNeckTest():
    """Base class to consolidate common logic for bottleneck layer tests."""

    LAYER_CLASS = None

    def setUp(self):
        """Set up."""
        self.input_shape = (1, 32, 32, 32, 16)
        self.filter_num = 32
        self.dropout_rate = 0.1
        self.stride = 2
        self.input_tensor = keras.random.normal(self.input_shape)

    def _build_layer(self, **kwargs):
        """Build a layer."""
        return self.LAYER_CLASS(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=self.stride,
            **kwargs
        )

    def _expected_shape(self, input_shape, stride, channel_mult):
        """Calculated the expected output shape"""
        scale = self._spatial_scale(stride)
        return (
            input_shape[0],
            int(input_shape[1] * scale),
            int(input_shape[2] * scale),
            int(input_shape[3] * scale),
            self.filter_num * channel_mult
        )

    def _spatial_scale(self, stride):
        """Not implemented."""
        raise NotImplementedError("Subclasses must implement _spatial_scale")

    def test_output_shape_and_type(self):
        """Output shapes tess.t"""
        for channel_mult in [4, 2, 1]:
            layer = self._build_layer(channel_out_mult_factor=channel_mult)
            output = layer(self.input_tensor)
            expected = self._expected_shape(self.input_shape, self.stride, channel_mult)

            self.assertEqual(output.shape, expected)
            self.assertEqual(layer.compute_output_shape(self.input_tensor.shape), expected)

    def test_model_compilation(self):
        """Model compile test."""
        inputs = keras.Input(shape=self.input_shape[1:])
        x = self._build_layer()(inputs)
        model = keras.Model(inputs, x)
        try:
            model.compile(optimizer='adam', loss='mse')
        except Exception as e:
            self.fail(f'Model compilation failed: {e}')

    def test_no_spatial_change(self):
        """Identity test."""
        layer = self.LAYER_CLASS(
            filter_num=self.filter_num,
            dropout_rate=self.dropout_rate,
            stride=1
        )
        output = layer(self.input_tensor)
        expected = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
            self.filter_num * layer.channel_out_mult_factor
        )
        self.assertEqual(output.shape, expected)

    def test_model_with_unknown_shapes(self):
        """Test unknown shapes on layer."""
        input_layer = keras.Input(shape=(None, None, None, 16))
        x = self._build_layer()(input_layer)
        x = keras.layers.GlobalAveragePooling3D()(x)
        output = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer="adam", loss="mse")

        dummy_input = keras.random.normal((1, 32, 32, 32, 16))
        dummy_target = keras.random.normal((1, 1))

        model.fit(dummy_input, dummy_target, epochs=1, steps_per_epoch=1, verbose=0)


# ===============================================
# *                BottleNeck                   *
# ===============================================
class TestBottleNeckLayer(BaseBottleNeckTest, unittest.TestCase):
    """Test BottleNeck"""
    LAYER_CLASS = layers.BottleNeck

    def _spatial_scale(self, stride):
        """Spatial scale is downsampled."""
        return 1 / stride  # Downsampling


# ===============================================
# *                UpBottleNeck                 *
# ===============================================
class TestUpBottleNeckLayer(BaseBottleNeckTest, unittest.TestCase):
    """Run tests for UpBottleNeckLayer."""
    LAYER_CLASS = layers.UpBottleNeck

    def _spatial_scale(self, stride):
        """Spatial scale is upsampled."""
        return stride  # Upsampling
    
# ===============================================
# *                    Plain                    *
# ===============================================
class TestPlainLayer(BaseBottleNeckTest, unittest.TestCase):
    """Run tests for Plain layer."""
    LAYER_CLASS = layers.Plain

    def _spatial_scale(self, stride):
        """Spatial scale is downsampled."""
        return 1 / stride  # Downsampling
    
# ===============================================
# *                    Residual                 *
# ===============================================
class TestResidualLayer(BaseBottleNeckTest, unittest.TestCase):
    """Run tests for Plain layer.."""
    LAYER_CLASS = layers.Residual

    def _spatial_scale(self, stride):
        """Spatial scale is downsampled."""
        return 1 / stride  # Downsampling
    
# ===============================================
# *                    Residual                 *
# ===============================================
class TestUpPlainLayer(BaseBottleNeckTest, unittest.TestCase):
    """Run tests for Plain layer.."""
    LAYER_CLASS = layers.UpPlain

    def _spatial_scale(self, stride):
        """Spatial scale is upsampled."""
        return stride  # Upsampling

# ===============================================
# *                    ExpandNeck               *
# ===============================================
class TestExpandNeckLayer(BaseBottleNeckTest, unittest.TestCase):
    """Run tests for Plain layer.."""
    LAYER_CLASS = layers.ExpandNeck

    def _spatial_scale(self, stride):
        """Spatial scale is downsampled.."""
        return 1 / stride  # Downsampling

# ===============================================
# *                SSFAdaLayer                  *
# ===============================================
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
        self.assertEqual(layer.compute_output_shape(input_tensor.shape), output.shape)

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
