from Network import DAE_NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy.stats as stats
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def get_layerwise_weight_dist(model):
    data, cols = [], []
    for l in model.layers:
        print('Layer:',l['name'])
        cols.append(l['name'])
        data.append( pd.Series(getattr(model,l['name']).weight.detach().numpy().flatten() ).describe() )
        print(stats.describe(getattr(model,l['name']).weight.detach().numpy(),axis=None))
        print('\n')
    print('\n')
    return pd.DataFrame(data,index=cols).reset_index().rename(columns={'index':'layer'})

# Within the init function of the neural network to be costructed we can call this to create the initialized layers


#def plot_grad_flow(named_parameters):
#    '''Plots the gradients flowing through different layers in the net during training.
#    Can be used for checking for possible gradient vanishing / exploding problems.
#    
#    Usage: Plug this function in Trainer class after loss.backwards() as 
#    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#    ave_grads = []
#    max_grads= []
#    layers = []
#    for n, p in named_parameters:
#        if(p.requires_grad) and ("bias" not in n):
#            layers.append(n)
#            ave_grads.append(p.grad.abs().mean())
#            max_grads.append(p.grad.abs().max())
#    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#    plt.xlim(left=0, right=len(ave_grads))
#    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#    plt.xlabel("Layers")
#    plt.ylabel("average gradient")
#    plt.title("Gradient flow")
#    plt.grid(True)
#    plt.legend([Line2D([0], [0], color="c", lw=4),
#                Line2D([0], [0], color="b", lw=4),
#                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#    #plt.show()

def plot_grad_flow(named_parameters,title):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.ylim(ymin=0,ymax=0.25)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow\n"+title)
    plt.grid(True)

train_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.flatten())])
)

test_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    download = True,
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.flatten())])
)

layer_params_2 = [
        {
            'name': 'fc1',
            'type': 0,
            'transformer': [nn.ReLU()],
            'out_features': 100
        },
        {
            'name': 'out',
            'type': 0,
            'transformer': [],
            'out_features': 10
        }
    ]

layer_params_1 = [
        {
            'name': 'fc1',
            'type': 0,
            'transformer': [nn.ReLU()],
            'out_features': 500
        },
        {
            'name': 'fc2',
            'type': 0,
            'transformer': [nn.ReLU()],
            'out_features': 500
        },
        {
            'name': 'fc3',
            'type': 0,
            'transformer': [nn.ReLU()],
            'out_features': 250
        },
        {
            'name': 'fc4',
            'type': 0,
            'transformer': [nn.ReLU()],
            'out_features': 100
        },
        {
            'name': 'out',
            'type': 0,
            'transformer': [],
            'out_features': 10
        }
    ]


def construct_model(layer_params_,train_set,flag,weight_decay=0):
    
    layer_params = copy.deepcopy(layer_params_)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1000)
    
    print('Data Loaded')
    
    if flag==0:
        # Prepare data
        print('Constructing Stacked Autoencoder Pretraining NN')
        model = DAE_NN.construct(layer_params,train_loader,weight_decay)
    elif flag==1:
        print('Constructing No Pretrainig NN')
        model = DAE_NN.construct_random(layer_params,train_loader)
    else:
        print('Constructing End-to-End Autoencoder Pretraining NN')
        model = DAE_NN.construct_simultaneous(layer_params,train_loader,weight_decay)
    
    print('********* Model Constructed *******\n\n')
    
    return model
    
def train_test(models,train_set):
    
    epoch, lr = 5, 0.01
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1000)
    optimizers = [ optim.Adam(m.parameters(), lr=lr) for _,m in models ]
    
#    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
#    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    
    print('\n**** Model Training ****\n')
    
    res = []
    
    plt.figure(figsize=(10,7))
    for j in range(epoch):
#        total_loss1, total_correct1 = 0, 0
#        total_loss2, total_correct2 = 0, 0        
        
        total_loss = [ 0 for _ in range(len(models))]
        total_correct = [ 0 for _ in range(len(models))]
        
        for batch in train_loader:
            images, labels = batch
            
            preds = [ m(images) for _, m in models ]
            loss = [ F.cross_entropy(p,labels) for p in preds ]
            
#            preds1 = model1(images)
#            loss1 = F.cross_entropy(preds1,labels)
#            
#            preds2 = model2(images)
#            loss2 = F.cross_entropy(preds2,labels)
            
            for i in range(len(models)):
                optimizers[i].zero_grad()
                loss[i].backward()
                plot_grad_flow(models[i][1].named_parameters(),models[i][0])
                optimizers[i].step()
                total_loss[i] += loss[i].item()
                total_correct[i] += preds[i].argmax(dim=1).eq(labels).sum()
                
#            optimizer1.zero_grad()
#            loss1.backward()
#            optimizer1.step()
#
#            optimizer2.zero_grad()
#            loss2.backward()
#            optimizer2.step()

            #print(preds.shape)
            #print(labels.shape)
            
#            total_loss1 += loss1.item()
#            total_correct1 += preds1.argmax(dim=1).eq(labels).sum()
#        
#            total_loss2 += loss2.item()
#            total_correct2 += preds2.argmax(dim=1).eq(labels).sum()
            
            
        
        for i in range(len(models)):
            res.append( [models[i][0],j,total_correct[i].item(),total_correct[i].item()/len(train_set),total_loss[i]/len(train_loader)] )
            print('model:',models[i][0],'\nepoch:',j,'\ntotal correct:',total_correct[i].item(),'\ntotal % correct:',total_correct[i].item()/len(train_set),'\nloss func:',total_loss[i]/len(train_loader))
        #print('epoch',i,'total correct:',total_correct2,'',total_loss2/len(train_loader))
        print('\n')
    
    return pd.DataFrame(res,columns=['Pretraining','Epoch','Total_correct','Total_correct_percentage','Loss_function'])
    
#model_init = construct_model(layer_params_2,train_set,0)
#model_random = construct_model(layer_params_2,train_set,1)
#model_autoencoder = construct_model(layer_params_1,train_set,2)

#models = [('DAE Layer-by-Layer',model_init),('Default Pytorch',model_random),('End-to-End Autoencoder',model_autoencoder)]
#models = [('End-to-End Autoencoder Pretraining',model_autoencoder)]
#models = [('No Pretraining Network',model_random)]
#models = [('Stacked Autoencoder Pretraining',model_init)]

#print('\n\n**** Layerwise weight distribution before training ****')
#get_layerwise_weight_dist(models[0][1])

#train_test(models,train_set)

'''
get_layerwise_weight_dist(model_init)
print('\n')
get_layerwise_weight_dist(model_random)
'''
#print('\n\n**** Layerwise weight distribution after training ****')
#get_layerwise_weight_dist(models[0][1])

model_list = [[1,'No Pretraining_0',0],
              [0,'Stacked Autoencoder Pretraining_0',0],
              [0,'Stacked Autoencoder Pretraining_00003',0.0002],
              [0,'Stacked Autoencoder Pretraining_00005',0.00025],
              [0,'Stacked Autoencoder Pretraining_00008',0.0003],
              [0,'Stacked Autoencoder Pretraining_0001',0.00035],
              [0,'Stacked Autoencoder Pretraining_0003',0.0004],
              [0,'Stacked Autoencoder Pretraining_01',0.00045]]

def train_2(model_list):

    w_after_total = []
    w_before_total = []
    train_res_total = []
    
    #for t, m_name in enumerate(['Stacked Autoencoder Pretraining','No Pretraining Network','End-to-End Autoencoder Pretraining']):
    for t, m_name, weight_decay in model_list:
        
        model = construct_model(layer_params_1,train_set,t,waeight_decay)
        models = [(m_name,model)]
        
        print('\n\n**** Layerwise weight distribution before training ****')
        w_before = get_layerwise_weight_dist(models[0][1])
        w_before['Pretraining'] = models[0][0]
        
        train_res = train_test(models,train_set)
        
        print('\n\n**** Layerwise weight distribution after training ****')
        w_after = get_layerwise_weight_dist(models[0][1])
        w_after['Pretraining'] = models[0][0]
        
        w_after_total.append(w_after)
        w_before_total.append(w_before)
        train_res_total.append(train_res)
    
    w_after_pd = pd.concat(w_after_total)
    w_before_pd = pd.concat(w_before_total)
    train_res_pd = pd.concat(train_res_total)
    
    return w_after_pd,w_before_pd, train_res_pd

def write_excel(w_after,w_before,train_res):

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('comparison.xlsx', engine='xlsxwriter')
    
    # Write each dataframe to a different worksheet. you could write different string like above if you want
    w_before.to_excel(writer, sheet_name='Sheet1')
    train_res.to_excel(writer, sheet_name='Sheet2')
    w_after.to_excel(writer,sheet_name='Sheet3')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

#res = train_2(model_list)
#write_excel(*res)