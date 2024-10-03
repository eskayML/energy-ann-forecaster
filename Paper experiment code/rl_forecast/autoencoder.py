from itertools import chain
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable


class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self, cnn_structure ,  kernel_size, padding=False,debug=False):
        """[summary]
        
        Arguments:
            cnn_structure {Array} -- Structure of the encoder side of the CNN, e.g. [16,12,8,4]
            kernel_size {int} -- [description]
        
        Keyword Arguments:
            adjust_groups {bool} -- [description] (default: {False})
            debug {bool} -- [description] (default: {False})
        """

        super(ConvolutionalAutoEncoder, self).__init__()
        self.debug = debug
        
        
        self.padding = kernel_size//2 if padding else 0
        
        self.encoder = nn.Sequential(OrderedDict([("cnn_layer{}_{}".format(input_size,output_size), 
                                                    nn.Conv1d(input_size,output_size,kernel_size,padding=self.padding))
                                                     for input_size,output_size in zip(cnn_structure[0:-1],cnn_structure[1:])]))
        self.decoder = nn.Sequential(OrderedDict([("deconv_layer{}_{}".format(input_size,output_size), 
                                                   nn.ConvTranspose1d(input_size,output_size,kernel_size,padding=self.padding))
                                                   for input_size,output_size in zip(reversed(cnn_structure[1:]),reversed(cnn_structure[0:-1]))]))
        
  
    def get_encoder(self,input):
        return self.encoder(input)

    def shape_print(self,name,output):
        if (self.debug):
            print("{}: {}".format(name,output.shape))
            
    def forward(self, input):
        encoder_out = self.encoder(input)
        return self.decoder(encoder_out)

class CnnMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(CnnMSELoss, self).__init__(None, None, "mean")

    def forward(self, input, target):
        return torch.mean(torch.sqrt(torch.mean(torch.mean(torch.pow((target-input),2),2),0)))

def init_net(network_structure, hidden_activation, use_batch_norm, include_activation_final_layer=True,dropout=None,
             combine_to_sequential=False):
    net = []

    # assures compatibility between single tasks and multitasks
    if not isinstance(network_structure, list):
        network_structure = [network_structure]

    for idx, x in enumerate(network_structure):
        sub_net = [x]
        if idx != len(network_structure) - 1 or include_activation_final_layer:
            if hidden_activation is not None:
                sub_net.append(hidden_activation)
            # as batch norm aims to "control" the input to the next layer
            # it makes sense to do it after the activation to take the actual value
            # including the non-linear activation into account
            # see to https://blog.paperspace.com/busting-the-myths-about-batch-normalization
            if use_batch_norm:
                sub_net.append(nn.BatchNorm1d(x.out_features))
            # https://forums.fast.ai/t/questions-about-batch-normalization/230
            if dropout is not None:
                    sub_net.append(nn.Dropout(dropout))
        if combine_to_sequential:
            net.append(nn.Sequential(*sub_net))
        else:
            net.extend(sub_net)

    return net

class Autoencoder(torch.nn.Module):

    def __init__(self, input_size, ann_structure, 
                hidden_activation=nn.LeakyReLU(negative_slope=0.1),\
                 use_batch_norm=False, final_layer_activation=None, 
                 dropout=None, embedding_module=None, embeding_position='start'):
        """
         input_size:
         ann_structure:
         hidden_activation:
         use_batch_norm:
         final_layer_activation:
         dropout:
         embedding_module: dies.mlp.Embedding, optional (default: None)
            Embeding layer for categorical data. 
         embeding_position: string, optional (default 'start')
            Position at which the output of the embedding layer is included. Either start or or bottleneck
        """

        super(Autoencoder, self).__init__()
        ann_structure = copy.copy(ann_structure)

        self.output_size = input_size

        self.embedding_module = embedding_module
        self.embeding_position = embeding_position

        if self.embedding_module is not None and self.embeding_position != 'start' \
            and self.embeding_position != 'bottleneck':
            raise ValueError("embeding_position must be start or bottleneck")

        if self.embedding_module is not None and self.embeding_position == 'start':
            input_size = input_size + self.embedding_module.no_of_embeddings

        ann_structure.insert(0, input_size)
        self.input_size = input_size
        
        self.network_structure_encoder = ann_structure.copy()
        self.network_structure_decoder = [ls for ls in reversed(ann_structure)]
        self.network_structure_decoder[-1] = self.output_size

        if self.network_structure_encoder[-1] != self.network_structure_decoder[0]:
            raise Warning('Last element of encoder should match first element of decoder.' + \
                        'But is {} and {}'.format(self.network_structure_encoder, self.network_structure_decoder))

        if self.embedding_module is not None and self.embeding_position == 'bottleneck':
            self.network_structure_decoder[0] = self.network_structure_decoder[0] + self.embedding_module.no_of_embeddings


        self.network_structure_encoder = [nn.Linear(x,y) for x,y in zip(self.network_structure_encoder[0:-1], self.network_structure_encoder[1:])]
        self.network_structure_decoder = [nn.Linear(x,y) for x,y in zip(self.network_structure_decoder[0:-1], self.network_structure_decoder[1:])]
        
        
        self.encoder = init_net(self.network_structure_encoder, hidden_activation, use_batch_norm,dropout=dropout)
        self.decoder = init_net(self.network_structure_decoder, hidden_activation, use_batch_norm, include_activation_final_layer=False,dropout=dropout)
        
        if final_layer_activation is not None: self.decoder.append(final_layer_activation)
        

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
        self.n_gpu = torch.cuda.device_count()   

       
    
    def forward(self, continuous_data, categorical_data=None):
        if self.embedding_module is not None and \
                        self.embeding_position == 'start':
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, continuous_data], 1) 
        else:
            x = continuous_data

        x = self.encoder(x)

        if self.embedding_module is not None and \
                        self.embeding_position == 'bottleneck':
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, x], 1) 

        x = self.decoder(x)
        
        return x




class AutoLSTM(torch.nn.Module):
    
    def __init__(self, autoencoder, nr_lstm_stages,output_size=1, hidden_size=5, levels_to_retrain=1, random_hidden_states=True):
        super(AutoLSTM, self).__init__()
        
        self.encoder = copy.deepcopy(autoencoder.encoder)
        self.nr_lstms = nr_lstm_stages
        self.encoder_fix_n_level = levels_to_retrain
        self.random_hidden_states = random_hidden_states
        self.lstm_hidden_size = hidden_size
        self.output_size = output_size
        # fix layers
        layers_fixed = 0
        encoder_output_size = 0
        for i,param in enumerate(self.encoder.parameters()):
            
            if layers_fixed < (len(self.encoder) - self.encoder_fix_n_level -1):
                param.requires_grad = False
            else:
                param.requires_grad = True
            encoder_output_size = param.shape[0]
            layers_fixed = layers_fixed + 1
        
        self.lstm = torch.nn.LSTM(encoder_output_size,num_layers=nr_lstm_stages,hidden_size=hidden_size,batch_first=True)

        self.linear = torch.nn.Linear(self.lstm_hidden_size,output_size)
        
    def forward(self,input):
        
        batch_size = input.shape[0]
        timesteps = input.shape[1]
        feature_size = input.shape[2]
        # iterate over sequences
        encoded_features = self.encoder(input.view(-1,feature_size))
        
        h0 = torch.zeros(self.nr_lstms, batch_size, self.lstm_hidden_size)
        c0 = torch.zeros(self.nr_lstms, batch_size, self.lstm_hidden_size)
        if (self.random_hidden_states):
            h0 = torch.randn(self.nr_lstms, batch_size, self.lstm_hidden_size)
            c0 = torch.randn(self.nr_lstms, batch_size, self.lstm_hidden_size)
        # put sequence thourgh lstm
        output, (hn, cn) = self.lstm(encoded_features.view(batch_size,timesteps,encoded_features.shape[1]),(h0,c0))


        fc_value = torch.functional.F.leaky_relu(self.linear(output))

        return fc_value
