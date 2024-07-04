"""
Created on 2024-07-02

@authors:
* Austin Dibble, University of Glasgow

Miscellaneous utility functions in brain library
"""

from tensorflow.keras.models import Model

def get_mid_layer(model, layer_name):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

def extract_latent_features(data, layer):
    """
    Extract latent features from a specific layer of the model.

    Parameters:
    - data: The input data to the model.
    - model: The pre-trained Keras model.
    - layer_name: The name of the layer from which to extract latent features.

    Returns:
    - latent_features: The extracted latent features.
    """
    latent_features = layer.predict(x=data, verbose=False)
    return latent_features