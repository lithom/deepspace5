import torch
from torch import nn
from torch.nn import Transformer
import torch.nn.functional as F
#from leet_deep.canonicalsmiles import LeetTransformer
import math

from deepspace5.architecture.basemodel import BaseModelConfiguration
from deepspace5.models.dataset_utils import pad_string_randomly


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# takes as input a sequence of ints
# assumes that output sequence is shorter than input sequence
class TransformerBlockA(nn.Module):
    def __init__(self, vocab_size_in, sequence_length_in, sequence_length_out, model_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerBlockA, self).__init__()
        self.embedding = nn.Embedding(vocab_size_in, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim,max_len=256)
        self.transformer = Transformer(d_model=model_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)#Transformer(d_model=model_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        self.sequence_length_in = sequence_length_in
        self.sequence_length_out = sequence_length_out
        self.linReducer = nn.Linear(sequence_length_in-sequence_length_out,sequence_length_out)

    def forward(self, input1, input2):
        # Embedding the inputs
        input1 = self.embedding(input1).permute(1, 0, 2)
        input2 = self.embedding(input2).permute(1, 0, 2)

        # Add positional encodings
        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        out_tf = self.transformer(input1, input2)

        out_a = out_tf.permute(1,0,2)

        out_formatted = out_a[:,:self.sequence_length_out,:]
        out_formatted_remainder = out_a[:, self.sequence_length_out:,:]
        out_formatted = F.relu( out_formatted + F.relu( self.linReducer( out_formatted_remainder.permute(0,2,1) ).permute(0,2,1) ) )

        return out_formatted




#@register_base_model("ExampleTransformerBaseModel")
class ExampleTransformerBaseModel(BaseModelConfiguration):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.global_data = None
        self.datafile = None
        self.datafile_smiles_input = None
        self.tokenized_data = None
        self.smiles_vocab = None
        self.padding_character = 'y'

    def load_base_model_from_config(self, config):
        self.model = TransformerBlockA(42,config['seq_length_in'],config['seq_length_out'],config['model_dim'],config['num_heads'],config['num_encoder_layers'],config['num_decoder_layers'])
        self.config = config
        self.datafile = self.config['datafile']
        self.datafile_smiles_input = self.config['datafile_smiles_input']

    def set_global_data(self, global_data):
        self.global_data = global_data
        self.tokenized_data = [xi[self.datafile_smiles_input] for xi in self.global_data[self.datafile]['Samples']]
        self.smiles_vocab = {char: idx + 1 for idx, char in enumerate(sorted(set(char for text in self.tokenized_data for char in text)))}
        if( self.padding_character in self.smiles_vocab ):
            print('[ERROR] padding character found in input smiles..')
        self.smiles_vocab[self.padding_character] = 0


    def create_model(self):
        return self.model

    def create_input_data_sample(self, idx: int):
        # Assume 'features' is a key in the loaded data dictionary
        smiles_raw = self.global_data[self.datafile]['Samples'][idx][self.datafile_smiles_input]
        smiles_raw_padded = pad_string_randomly(smiles_raw,self.config['seq_length_in'],pad_char=self.padding_character)
        smiles_encoded = torch.tensor(  [ self.smiles_vocab[ci] for ci in smiles_raw_padded ] )
        sample = {}
        sample["in"] = smiles_encoded
        return sample


def register_models(basemodels, outputhead_models):
    basemodels["TransformerBlockA"] = TransformerBlockA



