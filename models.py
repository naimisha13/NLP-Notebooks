from sentiment_data import *
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random


# The RNN model class.
class RNNet(nn.Module):
    def __init__(self, lstm_size, hidden_size, lstm_layers, drop_out, class_num, 
                 embedding, use_average, bidirectional, use_gpu=False):
        super(RNNet, self).__init__()
        self.feature_size = embedding.get_embedding_length()
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.g = torch.tanh   # fcc hidden layer activation function.
        self.drop_out = drop_out
        self.class_num = class_num
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.use_average = use_average
        
        # Set tensor type when using GPU or CPU
        if use_gpu:
            self.float_type = torch.cuda.FloatTensor
        else:
            self.float_type = torch.FloatTensor
        
        # The LSTM: forward sequence
        self.lstm1 = nn.LSTM(self.feature_size, self.lstm_size, self.lstm_layers)
        # The LSTM: backward sequence
        self.lstm2 = nn.LSTM(self.feature_size, self.lstm_size, self.lstm_layers)
        
        # The fully connected layer:
        if self.bidirectional:
            self.linear1 = nn.Linear(2*self.lstm_size, self.hidden_size)
        else:
            self.linear1 = nn.Linear(self.lstm_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.class_num)
        
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        
    # Initialize the hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (torch.zeros((self.lstm_layers, batch_size, self.lstm_size), 
                            requires_grad=False).type(self.float_type),
                torch.zeros((self.lstm_layers, batch_size, self.lstm_size), 
                            requires_grad=False).type(self.float_type))
    
    # Forward propagation to compute the states at each position.
    def forward(self, feats, feats_rev, seq_lens):
        batch_size = feats.size()[0]
        
        # Process forward sequence
        
        # (sequence length * batch size * feature size)
        feats = feats.transpose(1, 0).type(self.float_type)
        # Initialization of hidden and cell state for LSTM: (hx, cx)
        hidden = self.init_hidden(batch_size)
        # LSTM hidden states: (sequence length * batch size * lstm size)
        lstm1_outs, _ = self.lstm1(feats, hidden)
        # LSTM hidden states: (batch size * sequence length * lstm size)
        lstm1_outs = lstm1_outs.transpose(0, 1)
        # Get hidden vector corresponding to sequence length, 
        ff_feats = torch.zeros(batch_size, self.lstm_size)
        for i in range(batch_size):
            # Get hidden vector for actual sequence length.
            sent_ht = lstm1_outs[i][0:seq_lens[i]]
            if self.use_average:
                # Average the hidden vectors.
                ff_feats[i] = torch.mean(sent_ht, 0)
            else:
                # Use last time step of hidden vectors.
                ff_feats[i] = sent_ht[-1]
        # input to fully connected network.
        input_feats = ff_feats.type(self.float_type)

        
        # process reverse sequence if bidirectional is specified.
        if self.bidirectional:
            feats_rev = feats_rev.transpose(1,0).type(self.float_type)
            hidden = self.init_hidden(batch_size)
            lstm2_outs, _ = self.lstm2(feats_rev, hidden)
            lstm2_outs = lstm2_outs.transpose(0, 1)
            # Get hidden vector corresponding to sequence length, 
            ff_feats_rev = torch.zeros(batch_size, self.lstm_size)
            for i in range(batch_size):
                # Get hidden vector for actual sequence length.
                sent_ht = lstm2_outs[i][0:seq_lens[i]]  
                if self.use_average:
                    # Average the hidden vectors.:
                    ff_feats_rev[i] = torch.mean(sent_ht, 0)
                else:
                    # Use last time step of hidden vectors.
                    ff_feats_rev[i] = sent_ht[-1]
            # Concatenate forward and backward hidden vector for input to fully connected neural network.
            input_feats = torch.cat((ff_feats, ff_feats_rev), dim=1).type(self.float_type)
        
        # Output of the fully connected layers
        fc_out = F.dropout(self.g(self.linear1(input_feats)), p=self.drop_out)
        fc_out = F.dropout(self.g(self.linear2(fc_out)), p=self.drop_out)
        fc_out = self.linear3(fc_out)
        
        return torch.sigmoid(fc_out)
            
def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad/truncate to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    if length > np_arr.shape[0]:
        result = np.zeros(length)
        result[0:np_arr.shape[0]] = np_arr
    else:
        result = np_arr[0:length]
    return result

# Form the input to the neural network.
def form_input(x):
    return torch.from_numpy(x).float()

# Get sentence embedding vector which is given by
# average of the embeddings of the words in the sentence.
def get_sent_vec(exs: List[SentimentExample], word_vectors: WordEmbeddings):
    """
    : param exs: List of SentimentExample objects
    : word_vectors: WordEmbedding object
    : return: numpy array of sentence's words embedding
    """
    ex_size = len(exs)
    embedding_size = word_vectors.get_embedding_length()
    embeddings_vec = np.array(word_vectors.vectors).astype(float)
    ex_vec = np.zeros((ex_size, embedding_size)).astype(float)
    for i, ex in enumerate(exs):
        sent_embed = embeddings_vec[ex.indexed_words]
        ex_vec[i] = np.mean(sent_embed, axis=0)
    return ex_vec

