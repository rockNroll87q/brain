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