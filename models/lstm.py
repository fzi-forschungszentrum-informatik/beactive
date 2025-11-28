"""LSTM model architecture"""
import torch
import torch.nn as nn


def force_cudnn_initialization():
    """Force initialization of cudnn to avoid initialization overhead during training"""
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev),
                               torch.zeros(s, s, s, s, device=dev))

class LSTMModel(nn.Module):
    """ LSTM architecture for raw acc -> ema """

    def __init__(self, input_shape, output_shape):
        super().__init__()

        if torch.cuda.is_available():
            force_cudnn_initialization()

        self.linear0 = nn.Linear(input_shape[1],128)
        self.lstm = nn.LSTM(input_size=128,hidden_size=64, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(64,output_shape[0])

    def forward(self, x):
        torch.cuda.empty_cache()

        x = self.linear0(x)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.linear1(x)

        return x
