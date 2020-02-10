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

class EncoderAttention(nn.Module):

  def __init__(self, hidden_dim):
    super(EncoderAttention, self).__init__()
    
    self._intra_encoder = True

    self.wh = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
    self.ws = nn.Linear(hidden_dim * 2, hidden_dim * 2)
    self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()


  @property
  def intra_encoder(self) -> bool:
    """
    intra_encoder getter
    """
    return self._intra_encoder

  @intra_encoder.setter
  def intra_encoder(self, is_enabled: bool) -> bool:
    """
    intra_encoder setter
    """
    self._intra_encoder = is_enabled

  def forward(self, decoder_hidden, encoder_hidden, encoder_padding, 
              sum_temporal) -> Tuple:

    # (1) Standard Attention
    #     Fully Connected Layer
    #     Add out encoder and out decoder hidden states
    #     Tanh Activation function  
    et = self.wh(encoder_hidden)
    dec = self.ws(decoder_hidden).unsqueeze(1)
    et = et + dec 
    et = self.tanh(et)
    et = self.v(et).squeeze(2)

    # (2) Optional: If inter-temporal encoder attention is enabled
    if self.intra_encoder:
      et_exp = torch.exp(et)
      if sum_temporal is None:
        et1 = et_exp
        sum_temporal = torch.FloatTensor(et.size()).fill_(1e-10) + et_exp
        sum_temporal = sum_temporal.to(device)
      else:
        et1 = et_exp / sum_temporal 
        sum_temporal = sum_temporal + et_exp
    else:
      et1 = self.softmax(et, dim=1)

    # (3) Add probability of zero to padded elements
    at = et1 * encoder_padding 
    norm = at.sum(1, keepdim=True)
    at = at / norm
    at = at.unsquueze(1)

    # (4) Calculate enocder context vector
    cte = torch.bmm(at, encoder_hidden)
    cte = cte.squeeze(1)
    at = at.squeeze(1)

    return cte, at, sum_temporal


