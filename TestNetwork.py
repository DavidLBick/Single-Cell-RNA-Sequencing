from Network import DAE_NN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy.stats as stats
import copy

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
            'transformer': [F.relu],
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
            'transformer': [F.relu],
            'out_features': 500
        },
        {
            'name': 'fc2',
            'type': 0,
            'transformer': [F.relu],
            'out_features': 500
        },
        {
            'name': 'fc3',
            'type': 0,
            'transformer': [F.relu],
            'out_features': 250
        },
        {
            'name': 'fc4',
            'type': 0,
            'transformer': [F.relu],
            'out_features': 100
        },
        {
            'name': 'out',
            'type': 0,
            'transformer': [],
            'out_features': 10
        }
    ]


def construct_model(layer_params_,train_set,flag):
    
    layer_params = copy.deepcopy(layer_params_)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1000)
    
    print('Data Loaded')
    
    if flag==0:
        # Prepare data
        print('Constructing DAE Initialized NN')
        model = DAE_NN.construct(layer_params,train_loader)
    elif flag==1:
        print('Constructing Random Initialized NN')
        model = DAE_NN.construct_random(layer_params,train_loader)
    else:
        print('Constructing Autoencoder Initialized NN')
        model = DAE_NN.construct_simultaneous(layer_params,train_loader)
    
    print('Model Constructed')
    
    return model
    
def train_test(models,train_set):
    
    epoch, lr = 5, 0.01
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1000)
    optimizers = [ optim.Adam(m.parameters(), lr=lr) for _,m in models ]
    
#    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
#    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    
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
            print('model',models[i][0],'epoch',j,'total correct:',total_correct[i].item(),'total % correct:',total_correct[i].item()/len(train_set),'loss func:',total_loss[i]/len(train_loader))
        #print('epoch',i,'total correct:',total_correct2,'',total_loss2/len(train_loader))
        print('\n')

model_init = construct_model(layer_params_1,train_set,0)
model_random = construct_model(layer_params_1,train_set,1)
model_autoencoder = construct_model(layer_params_1,train_set,2)

models = [('DAE Layer-by-Layer',model_init),('Default Pytorch',model_random),('End-to-End Autoencoder',model_autoencoder)]
#models = [('End-to-End Autoencoder',model_autoencoder)]

train_test(models,train_set)

def get_layerwise_weight_dist(model):
    for l in model.layers:
        print('Layer:',l['name'])
        print(stats.describe(getattr(model,l['name']).weight.detach().numpy(),axis=None))

'''
get_layerwise_weight_dist(model_init)
print('\n')
get_layerwise_weight_dist(model_random)
'''