import torch
from torch import nn
from network import CNN
import numpy as np

class feature_extractor:
    """Extract activation and gradient from intermediate layers of neural networks.
    
    Attributes
    ----------
    model : nn.model
        Trained network
    module : nn.module
        Neural network module to register hook on
        
    """

    def __init__(self, model, module):
        self.model = model
        self.module = module
        self.outputs = {}

        module.register_forward_hook(self.get_activation())
        module.register_full_backward_hook(self.get_gradient())

    def get_activation(self):
        """Get activation map from network layer with hooks
        
        Returns
        -------
        function
            hook function to extract activations
            
        """
        
        def hook(network, input, output):
            self.outputs['activation'] = output.clone().detach()
        return hook

    def get_gradient(self):
        """Get gradient map from network layer with hooks
        
        Returns
        -------
        function
            hook function to extract gradients
            
        """
        def hook(network, grad_input, grad_output):
            self.outputs['gradient'] = grad_output
        return hook

    def get_features(self, x, index):
        """Call to get activations and gradients at an intermediate network layer
        
        Parameters
        ----------
        x : torch.tensor
            Data
        index : int
            Class to get predictions for
          
        Returns
        -------
        dictionary
            activation map
        torch.tensor
            gradients for specific class
                
        """
        m = nn.Softmax()
        I = x.clone().detach()
        prediction = m(self.model(I))

        score = prediction[:, index]
        self.model.zero_grad()
        score.backward()
        
        pred = True
        if torch.abs(prediction[:, 0][0]) >= torch.abs(prediction[:, 1][0]):
            pred = False
        

        return self.outputs['activation'], self.outputs['gradient'], pred
