import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Within the init function of the neural network to be costructed we can call this to create the initialized layers

def apply(x,funcs):
    for f in funcs:
        x = f(x)
    return x

class LinearAutoencoder(nn.Module):
    
    def __init__(self, in_features, hidden_layer_features, op_transformer ):
        
        super(LinearAutoencoder, self).__init__()
        self.in_features = in_features
        self.hidden_layer_features = hidden_layer_features
        self.op_transformer = op_transformer
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layer_features)
        self.out = nn.Linear(in_features=hidden_layer_features, out_features=in_features)
        
    def forward(self,X):
        
        # Encoding: Input to hidden layer
        t = self.fc1(X)
        t = apply(t,self.op_transformer)
        
        # Decoding: Hidden to Output layer
        t = self.out(t)
        
        return t

# Class containing factory methods to construct initialized-layers
class Autoencoder:
    
    @staticmethod
    def add_noise(X):
        return X*torch.tensor(np.random.choice([0,1],size=X.shape,replace=True, p=[0.6,0.4]))
    
    @staticmethod
    def get_linear_infeatures(dataloader,feature_transformer):
        X, _ = next(iter(dataloader))
        
        # FIXME
        in_features = len(feature_transformer(X[0]))
        return in_features
    
    @staticmethod
    def combine_transformers(fc,feature_transformer,op_transformer):
        return lambda x: apply(fc(feature_transformer(x)),op_transformer)
    
    @staticmethod
    def get_linear_layer_random(dataloader,hidden_layer_features,op_transformer,feature_transformer=lambda x: x):
        
        in_features = Autoencoder.get_linear_infeatures(dataloader,feature_transformer)
        fc = nn.Linear(in_features=in_features,out_features=hidden_layer_features)
        
        return fc, Autoencoder.combine_transformers(fc,feature_transformer,op_transformer), 0.0 
    
    @staticmethod
    def get_linear_layer(dataloader,hidden_layer_features,op_transformer,feature_transformer=lambda x: x):
        
        epoch = 5
        lr = 0.01
        
        in_features = Autoencoder.get_linear_infeatures(dataloader,feature_transformer)
        
        model = LinearAutoencoder(in_features,hidden_layer_features,op_transformer)
        
        optimizer = optim.Adam(model.parameters(),lr=lr)
        loss_func = F.mse_loss        
        
        total_loss = Autoencoder.train(model,loss_func,optimizer,epoch,dataloader,feature_transformer)
        fc = model.fc1
        
        return fc, Autoencoder.combine_transformers(fc,feature_transformer,op_transformer), total_loss
    
    @staticmethod
    def train_autoencoder(model,dataloader):
        lr=0.01
        loss_func = F.mse_loss
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1)
        epoch = 5
        feature_transformer = lambda x: x
        
        Autoencoder.train(model,loss_func,optimizer,epoch,dataloader,feature_transformer)
        
    @staticmethod
    def train(model, loss_func, optimizer, epoch, dataloader, feature_transformer):
        total_loss = np.zeros(epoch)
        
        for i in tqdm(range(epoch)):
            
            epoch_loss = 0    
            
            for X, _ in dataloader:
                
                noisy_X = Autoencoder.add_noise(feature_transformer(X))
                #print(noisy_X.shape)
                
                preds = model(noisy_X)
                
                # May need to be changed for 
                loss = loss_func(preds,feature_transformer(X))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss[i] = epoch_loss
            print('For epoch',i,'MSE loss is',epoch_loss)
        
        return total_loss

'''
Input: Provide a list of dict D

Each dict D should contain the following keys only:
    
1. layer_type: Int [0:Linear, 1:Convolution]
2. layer_name
3. transformer: List of functions applied on the layer. These should not contain learnable parameters

Corresponding to the layer type, provide relevant keys:

LINEAR LAYER
1. in_features
2. out_features

Output: Constructs a DAE-based initialized Neural Net
'''

class DAE_NN(nn.Module):
    
    def __init__(self, layers):
        
        super(DAE_NN, self).__init__()
        self.layers = layers
        for l in layers:
            setattr(self,l['name'],l['obj'])
    
    def forward(self,X):
        
        for l in self.layers:
            X = getattr(self,l['name'])(X)
            for f in l['transformer']:
                X = f(X)
    
        return X
    
    @staticmethod
    def construct_simultaneous(layer_params,dataloader):
        #layer_params = layer_params_.copy()
        layers = DAE_NN.construct_random_helper(layer_params,dataloader)
        
        layers[-1]['transformer'] = [F.relu]
        extra_layer = {'name':'out_extra',
                       'type':0,
                        'transformer': [],
                        'out_features': layers[0]['obj'].in_features,
                        'obj': nn.Linear(in_features=layers[-1]['out_features'],out_features=layers[0]['obj'].in_features )
                       }
        
        layers.append(extra_layer)
        #print('*******')
        #print(layers)
        temp_model = DAE_NN(layers)
        Autoencoder.train_autoencoder(temp_model,dataloader)
        
        for l in layers:
            l['obj'] = getattr(temp_model,l['name'])
        
        layers[-2]['transformer'] = []
        
        return DAE_NN(layers[:-1])
        
    @staticmethod
    def construct_random_helper(layer_params,dataloader):
        
        next_transformer = lambda x: x
        
        for l in layer_params:
            
            init_layer, next_transformer, init_loss = Autoencoder.get_linear_layer_random(dataloader,l['out_features'],l['transformer'],feature_transformer=next_transformer)
            l['obj'] = init_layer
        
        return layer_params

    @staticmethod
    def construct_random(layer_params,dataloader):
        #layer_params = layer_params_.copy()
        return DAE_NN(DAE_NN.construct_random_helper(layer_params,dataloader))
    
    @staticmethod
    def construct(layer_params,dataloader):
        #layer_params = layer_params_.copy()
        next_transformer = lambda x: x #.flatten(start_dim=1)
        
        for l in layer_params:

            print('Training Layer',l['name'],': Begin')
            init_layer, next_transformer, init_loss = Autoencoder.get_linear_layer(dataloader,l['out_features'],l['transformer'],feature_transformer=next_transformer)
            print('Training Layer',l['name'],': End')
            l['obj'] = init_layer
        
        return DAE_NN(layer_params)
        