"""
Created on Tuesday - September 03 2024

@authors:
* Austin Dibble, University of Glasgow

Utility functions for running LRP. 
Note this file assumes that keras-explainability/explainability is in the python path.

Usage example:

    lrp_config = LRPConfig.build_default_config()
    lrp_model = LRPModelFactory(model, lrp_config, 'PatientAge', output_idx=0).build()

"""
from loguru import logger

try:
    from explainability import LRP, LRPStrategy
    from explainability.layers import SSFLRP, BottleNeckLRP, ConvLRP, DenseLRP
except ImportError as e:
    logger.error(f"Could not import explainability: {e}")

from keras.models import Model

from .model_utils import freeze_model


class LRPStrategyBuilder:
    """
    Build a LRPStrategy object automagically, given the input parameters. 
    From what I can tell, alpha & beta apply to conv layers. Epsilon is for dense. 
    B layers are for inputs? 
    I haven't investigated that thoroughly. My defaults seem to work.

    Args:
    - model: input brain age model. Should work with just about any model though I think.
    - alpha: float parameter to focus positive predictions.
    - beta: float parameter to focus negative predictions.
    - epsilon: float parameter to .... ?
    - num_b_layers: The number of beginning layers on which to apply the 'b' parameter. 1-2 seems to work.
    """
    def __init__(self, model:Model, alpha:float=1, beta:float=1, epsilon:float=0.25, num_b_layers:int=1):
        """Initialise"""
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.num_b_layers = num_b_layers
        self.strategy_layers = []

    @staticmethod
    def _get_conv_layer_strategy(alpha, beta):
        """Basic conv layer strategy dict."""
        return {'alpha': alpha, 'beta': beta}
    
    @staticmethod
    def _get_dense_layer_strategy(epsilon):
        """Basic dense layer strategy dict."""
        return {'epsilon': epsilon}
    
    @staticmethod
    def _get_input_layer_strategy(b, flat):
        """Basic input layer strategy dict."""
        return {'b': b, 'flat': flat}

    def _get_layer_strategy(self, layer):
        """Get the appropriate layer parameters given the LRP layer required."""
        if isinstance(layer, ConvLRP):
            return self._get_conv_layer_strategy(self.alpha, self.beta)
        elif isinstance(layer, DenseLRP):
            return self. _get_dense_layer_strategy(self.epsilon)
        elif isinstance(layer, BottleNeckLRP):
            return self._get_conv_layer_strategy(self.alpha, self.beta)
        elif isinstance(layer, SSFLRP):
            return self._get_dense_layer_strategy(self.epsilon)
            
        return None

    def build(self):
        """Builds the strategy object based on the configuration given in this class."""
        tmp_lrp = LRP(self.model, layer=len(self.model.layers)-1, idx=0) # build LRP with no strategy to infer layers
    
        # first conv are set to ['b': True, 'flat': True]. 
        # Why? ¯\_(ツ)_/¯
        # The paper mentions the 'b' parameter in their input and first conv.
        # In their example notebook, they don't use b, but they do use 'flat'. 
        # I've tried both. Seems to work? Smh.
        self.strategy_layers += [self._get_input_layer_strategy(b=True, flat=True)] * self.num_b_layers
        idx = 0
        for layer in (reversed(tmp_lrp.layers)):
            s_layer = self._get_layer_strategy(layer)
            if s_layer is not None:
                idx += 1
                if idx >= 1 + self.num_b_layers: # Skip model's first CNN layer as it's already been added
                    logger.debug(f'{idx} - {layer.name}:{s_layer}')
                    self.strategy_layers.append(s_layer)

        return LRPStrategy(layers=self.strategy_layers)


class LRPConfig:
    """Simple object class for storing LRP configuration parameters."""
    def __init__(self, alpha:float=1, beta:float=1, epsilon:float=0.25, num_b_layers:int=2):
        """Initialise object with params."""
        self.config = {'alpha': alpha, 'beta': beta, 'epsilon': epsilon, 'num_b_layers': num_b_layers}

    def get_params(self):
        """Get params."""
        return self.config

    @staticmethod
    def build_default_config():
        """Build new object based on default/good enough."""
        return LRPConfig(alpha=2, beta=1, epsilon=0.25, num_b_layers=2)

class LRPModelFactory:
    """Factory class which builds the new LRP model based on the given configuration parameters."""
    def __init__(self, model, lrp_config:LRPConfig, lrp_variable:str, output_idx:int=0):
        """Init"""
        self.model = model
        self.lrp_config = lrp_config
        self.lrp_variable = lrp_variable
        self.output_idx = output_idx

    def build(self):
        """Build the LRP model object."""
        output_layer = None

        model_outputs = self.model.output
        if not isinstance(model_outputs, list | tuple):
            model_outputs = [model_outputs]

        for out in model_outputs:
            if self.lrp_variable in out.name:
                output_layer = out

        if output_layer is None:
            raise ValueError(f"Expected output variable {self.lrp_variable} not found in the source model's outputs.")

        new_model = Model(self.model.input, output_layer)
        freeze_model(new_model)

        strategy_builder = LRPStrategyBuilder(new_model, **self.lrp_config.get_params())
        strategy = strategy_builder.build()

        return LRP(new_model, layer=len(new_model.layers) - 1, strategy=strategy, idx=self.output_idx)
