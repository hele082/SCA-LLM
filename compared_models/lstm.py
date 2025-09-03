import torch
import torch.nn as nn

from utils.model_io import capture_init_args


class LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, pred_len: int, features_size: int, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        """
        Initializes the LSTM model.

        Args:
            pred_len (int): Prediction length (number of future steps to predict).
            features_size (int): The number of features in the input and output sequences.
            input_size (int): The dimensionality of the input fed into the LSTM layers
                                (often same as features_size after an initial projection).
            hidden_size (int, optional): The number of features in the hidden state h. Defaults to 256.
            num_layers (int, optional): Number of recurrent layers. Defaults to 2.
        """
        super(LSTM, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.pred_len = pred_len
        self.features_size = features_size
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.model = LSTMUnit(features_size=self.features_size, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        
    def forward(self, x):
        device = x.device
        batch_size, seq_len, features_size = x.shape
        prev_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        prev_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(),
                                                            prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output, prev_hidden, prev_cell)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs
    

class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features_size, input_size, hidden_size=256, num_layers=2):
        super(LSTMUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features_size, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features_size))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden, cur_cell

