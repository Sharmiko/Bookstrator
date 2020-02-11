import torch 
import torch.nn as nn  
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Tuple

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Dense Layers
    self.wh = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
    self.ws = nn.Linear(hidden_dim * 2, hidden_dim * 2)
    self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    # Activations Functions
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


class DecoderAttention(nn.Module):

  def __init__(self, hidden_dim):
    super(DecoderAttention, self).__init__()

    self._intra_decoder = True

    # If intra decoder is enabled initialize layers
    if self.intra_decoder:
      self.w_prev = nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.ws = nn.Linear(hidden_dim, hidden_dim)
      self.v = nn.Linear(hidden_dim, 1, bias=False)

    # Activation Functions 
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  @property
  def intra_decoder(self) -> bool:
    """
    intra_decoder getter
    """
    return self._intra_decoder

  @intra_decoder.setter
  def intra_decoder(self, is_enabled: bool) -> bool:
    """
    intra_decoder setter
    """
    self._intra_decoder = is_enabled

  def forward(self, hidden_state, attention):

    # If Intra decoder is not enabled
    # create zeros with same dim as hidden state
    if not self.intra_decoder:
      ctd = torch.zeros(hidden_state.size()).to(device)
    # If no attention was provided
    elif attention is None:
      ctd = torch.zeros(hidden_state.size()).to(device)
      attention = hidden_state.unsquueze(1)
    # Standard Attention
    # Fully Connected Layer
    # Add out encoder and out decoder hidden states
    # Tanh Activation function 
    else:
      et = self.w_prev(attention)
      fea = self.ws(hidden_state).unsquueze(1)
      et = et + fea
      et = self.tanh(et)
      et = self.v(et).squueze(2)

      # Decoder attention
      at = self.softmax(et, dim=1).unsqueeze(1)
      ctd = torch.bmm(at, attention).squeeze(1)
      attention = torch.cat([attention, hidden_state.unsquueze(1)], dim = 1)

    return ctd, hidden_state

class Decoder(nn.Module):

  def __init__(self, hidden_dim, embed_dim, vocab_size):
    super(Decoder, self).__init__()

    # Attention Layers
    self.encoder_attention = EncoderAttention(hidden_dim)
    self.decoder_attention = DecoderAttention(hidden_dim)

    # Context 
    self.context = nn.Linear(hidden_dim * 2 + embed_dim, embed_dim)

    # LSTM Layer
    self.lstm = nn.LSTMCell(embed_dim, hidden_dim)

    # Dense Layer
    self.fc = nn.Linear(hidden_dim * 5 + embed_dim, 1)

    self.v1 = nn.Linear(hidden_dim * 4, hidden_dim)
    self.v2 = nn.Linear(hidden_dim, vocab_size)

    # Activation Functions
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()

  def forward(self, x_t, s_t, encoder_out, encoder_padding, cte, zeros, 
              encoder_batch_extend, sum_temporal, prev_state):
    
    t = self.context(torch.cat([x_t, cte], dim=1))
    s_t = self.lstm(t, s_t)

    decoder_hidden, decoder_cell = s_t 
    st_hat = torch.cat([decoder_hidden, decoder_cell], dim=1)
    cte, attention, sum_temporal = self.encoder_attention(st_hat, encoder_out,
                                                          encoder_padding.
                                                          sum_temporal)
    ctd, prev_state = self.decoder_attention(decoder_hidden, prev_state)

    gen = torch.cat([cte, ctd, st_hat, t], 1)
    gen = self.fc(gen)
    gen = self.sigmoid(gen)

    out = torch.cat([decoder_hidden, cte, ctd], dim=1)
    out = self.v1(out)
    out = self.v2(out)

    distribution = self.softmax(out, dim=1)
    distribution = gen * distribution

    attention = (1 - gen) * attention

    if zeros is not None:
      distribution = torch.cat([distribution, zeros], dim=1)
    distribution = distribution.scatter_add(1, encoder_batch_extend, attention)

    return distribution, s_t, cte, sum_temporal, prev_state
    
class TextSummarizer(nn.Module):

  def __init__(self, input_size, hidden_size, embed_size, vocab_size):
    super(TextSummarizer, self).__init__()

    # Encoder, Decoder, Embed Layer
    self.encoder = Encoder(input_size, hidden_size)
    self.decoder = Decoder(hidden_size, embed_size, vocab_size)
    self.embed = nn.Embedding(vocab_size, embed_size)

     # Convert models into appropriate device
    self.encoder = self.encoder.to(device)
    self.decoder = self.decoder.to(device)
    self.embed = self.embed.to(device)