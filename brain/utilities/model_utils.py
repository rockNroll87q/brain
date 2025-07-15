"""
Created on 2025-02-06

@authors:
* Austin Dibble, University of Glasgow

Utility functions for manipulating models and model layers.
"""

from keras.models import Model
from loguru import logger

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


def print_model_layer_names(model, print_details=True):
    """
    Print all the layer names from the TF/keras model. 
    Lists if each layer is trainable or not.

    Args:
        model: TF/keras model
        print_details: print out inner parts of the layer.
    """
    logger.info('Printing model layer names:')
    for layer in model.layers:
        logger.info(f'{layer.name} -- Trainable: {layer.trainable}')
        if print_details:
            for var in layer.variables:
                logger.info(f'\t{var.name}')

def freeze_level(model, level: int = 0):
    """
    Freeze a given coarse-to-fine level
    :param model: tf model
    :param level: level name
    """
    model.layers[level].trainable = False


def freeze_level_lte(model, level_lte: int = 0):
    """
    Freeze coarse-to-fine levels up to a given level
    :param model: tf model
    :param level_lte: level less than or equal
    """
    for k in range(level_lte+1):
        freeze_level(model, k)

def freeze_level_mt(model, level_mt: int = 0):
    """
    Freeze coarse-to-fine levels past a given level
    :param model: tf model
    :param level_mt: level more than this will be frozen
    """
    for k in range(level_mt,len(model.layers)):
        freeze_level(model, k)

def freeze_model(model):
    """
    Free all the layers in a given model
    :param model: tf model
    """
    for layer in model.layers:
        layer.trainable = False

def set_trainable(model):
    """
    Set all the layer in a given model to trainable
    :param model: tf model
    """
    for layer in model.layers:
        layer.trainable = True
    # recompile(model) Bug: raise an error if tf==2.4.1. Is it needed?


def set_trainable_after_layer(model, target_layer_name):
    """
    Sets all layers after the specified layer to be trainable.
    
    Args:
        model (tf.keras.Model): The model to modify.
        target_layer_name (str): The name of the layer after which all layers should be trainable.
    """
    found = False
    for layer in model.layers:
        if found:
            layer.trainable = True
        elif layer.name == target_layer_name:
            found = True  # Start setting layers as trainable from the next layer

    return found

def find_layer(model, target_layer_name):
    """
    Finds the layer using the given string name. If it's not found, returns layer index of -1.
    
    Args:
        model (tf.keras.Model): The model to modify.
        target_layer_name (str): The name of the layer to find.

    Returns:
        (layer index, layer object)
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == target_layer_name:
            return idx, layer

    return -1, None