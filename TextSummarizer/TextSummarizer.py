import torch 
import torch.nn as nn  
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Tuple

class Encoder(nn.Module):

  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()

    # LSTM layer - (input_size, hidden_size)
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                        batch_first=True, bidirectional=True)

    # Fully connected layers for hidden and cell states
    self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
    self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    # ReLU Activation function
    self.relu = nn.ReLU()

  def forward(self, t, seq_len) -> Tuple:

    # (1) Input Layer
    t = t

    # (2) Pack a tensor containing padded sequence
    packed = pack_padded_sequence(t, seq_len, batch_first=True)

    # (3) Hidden LSTM Layer
    t, hidden = self.lstm(packed)

    # (4) Pad a packed length sequence
    t, _ = pad_packed_sequence(t, batch_first=True)

    # (5) Make sure tensor is stored in contiguous chunk of memory
    #     If tensor if contiguous nothing is changed
    t = t.contiguous()

    # (6) Unpack hidden state into -> hidden state and cell state
    h, c = hidden

    # (7) Concatenate hidden state along dim=1
    #     Fully Connected Layer
    #     ReLU Activation Function
    h = torch.cat(list(h), dim=1)
    h = self.fc_h(h)
    h = self.relu(h)

    # (8) Concatenate cell state along dim=1
    #     Fully Connected Layer
    #     ReLU Activation Function
    c = torch.cat(list(c), dim=1)
    c = self.fc_c(c)
    c = self.relu(c)

    # return tensor, hidden and cell state
    return t, (h, c)
