"""GRU model architecture"""
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.gru = nn.GRU(input_size=input_shape[1],hidden_size=50, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(50,output_shape[0])

    def forward(self, x):
        """Forward pass of the model"""
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.linear1(x)
        return x
