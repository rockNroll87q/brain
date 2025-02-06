"""
Created on 2024-07-02

@authors:
* Austin Dibble, University of Glasgow

Miscellaneous utility functions in brain library
"""

from tensorflow.keras.models import Model
import numpy as np

def get_mid_layer(model, layer_name):
    """
    Returns a new Keras model that has the outputs as the specified intermediate layer.
    This can be useful for extracting features or using the output of a specific 
    layer. Pass the returned object of this function to `extract_latent_features`, 
    for example.

    Parameters:
        - model (Keras model): The original model from which to extract
            the intermediate layer.
        - layer_name (str): The name of the intermediate layer to 
            extract.

    Returns:
        - intermediate_layer_model (Keras model): A new Keras model that 
            only goes up to the specified intermediate layer.
    """
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

def extract_latent_features(data, layer):
    """
    Extract latent features from a specific layer of the model.

    Parameters:
    - data: The input data to the model.
    - model: The pre-trained Keras model.
    - layer: The layer object from which to extract latent features. Can be obtained from `get_mid_layer`.

    Returns:
    - latent_features: The extracted latent features.

    Example::

        # Get latent features from test data
        try:
            model_layer = get_mid_layer(model, layer_name=layer_name)
        except Exception as e:
            # Handle error
            return

        gt_latent_vectors = extract_latent_features(test_data, model_layer)
    """
    latent_features = layer.predict(x=data, verbose=False)
    return latent_features

def build_model_connectivity_graph(model):
    """
    Builds a nested dictionary representation of the connectivity graph of the given model.
    This can then be used with other functions. 

    Params:
    - model: tf/keras model object

    Returns:
    - graph object. Each layer name has a dict with 'inbound' and 'outbound' layers as their string names.
    """
    graph = {layer.name: {'inbound': [], 'outbound': []} for layer in model.layers}
    for layer in reversed(model.layers):
        inbound_layers = []
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                inbound_layers += [inbound_layer.name for inbound_layer in node.inbound_layers]
            else: 
                inbound_layers.append(node.inbound_layers.name)

            for inbound in inbound_layers:
                graph[inbound]['outbound'].append(layer.name) # Add this layer as an outbound link
                
        graph[layer.name]['inbound'] = inbound_layers # All get added as inbound links
        
    return graph

def remove_model_connectivity(model):
    """
    Delete the internal node connectivity of the model. This is necessary when changing 
    model by inserting layers.
    """
    for layer in model.layers:
        layer._inbound_nodes = []


def pad_volume_to_shape(volume, target_shape=(256,256,256)):
    # padding = [(0, max(target_dim - current_dim, 0)) for target_dim, current_dim in zip(target_shape, volume.shape)]

    padding = []
    for current_dim, target_dim in zip(volume.shape, target_shape):
        total_padding = target_dim - current_dim
        # Split padding evenly between 'before' and 'after', with extra padding added to 'after' if odd
        before_padding = total_padding // 2
        after_padding = total_padding // 2 + total_padding % 2  # Add the extra padding to 'after' if odd
        padding.append((before_padding, after_padding))

    padded_vol = np.pad(volume, padding, mode='constant', constant_values=0)
    return padded_vol, padding