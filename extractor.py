import torch
from torch import nn
from network import CNN


class feature_extractor:
    """
    Extracts activation and gradient from 
        intermediate layers of neural networks with hooks
    """

    def __init__(self, model, module):
        """
        :param model: trained network
        :param module: neural network module to register hook on
        :return: tensors of activation and gradient values
        """
        self.model = model
        self.module = module
        self.outputs = {}

        module.register_forward_hook(self.get_activation())
        module.register_full_backward_hook(self.get_gradient())

    def get_activation(self):
        def hook(network, input, output):
            self.outputs['activation'] = output.clone().detach()
        return hook

    def get_gradient(self):
        def hook(network, grad_input, grad_output):
            self.outputs['gradient'] = grad_output
        return hook

    def get_features(self, x, index):
        m = nn.Softmax()
        I = x.clone().detach()
        prediction = m(self.model(I))

        score = prediction[:, index]
        self.model.zero_grad()
        score.backward()

        return self.outputs['activation'], self.outputs['gradient']
