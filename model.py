"""
The model class.
"""
import torch
import torch.nn as nn
import torchvision.models as models

HIDDEN_LAYER_SIZE = 1024


class KeypointPredictionModel(nn.Module):
    def __init__(self):
        super(KeypointPredictionModel, self).__init__()
        # ResNet w/o the last layer
        resnet50 = models.resnet50(pretrained=False)
        self.resnet_mod = nn.Sequential(*list(resnet50.children())[:-1])
        # Fully connected layers
        self.fully_connected1 = nn.Linear(2048, HIDDEN_LAYER_SIZE)
        self.fully_connected2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fully_connected3 = nn.Linear(HIDDEN_LAYER_SIZE, 42)

    def forward(self, x):
        resnet_mod_out = self.resnet_mod(x)
        fc_in = torch.flatten(resnet_mod_out, start_dim=1)
        fc_h1 = self.fully_connected1(fc_in)
        fc_h2 = self.fully_connected2(fc_h1)
        out = self.fully_connected3(fc_h2)
        return out
