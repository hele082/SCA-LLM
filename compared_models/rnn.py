import torch
import torch.nn as nn

from utils.model_io import capture_init_args


class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, pred_len, feature_size, input_size, hidden_size=256, num_layers=2):

        super(RNN, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.pred_len = pred_len
        self.features_size = feature_size
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.model = RNNUnit(features_size=self.features_size, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        
    def forward(self, x):
        device = x.device
        batch_size, seq_len, n_features = x.shape
        prev_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs
    

class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features_size, input_size, hidden_size=256, num_layers=2):
        super(RNNUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features_size, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features_size))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden