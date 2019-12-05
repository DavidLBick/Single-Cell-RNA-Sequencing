import torch
import torch.nn as nn
import torch.nn.functional as F
import config 
import pdb


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

class BaselineModel(nn.Module):
    def __init__(self, input_size, classes):
        super(BaselineModel, self).__init__()
        self.embedding_model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000), 
            nn.ReLU(),
            nn.Linear(1000, 1000), 
            nn.ReLU(),
            nn.Linear(1000, 1000), 
            nn.ReLU(),
            nn.Linear(1000, 1000), 
            nn.ReLU(),
            nn.Linear(1000, 100))

        # final_hidden_size = set here output size of self.embedding_model 
        self.classification_model = nn.Sequential(
            nn.ReLU(True), 
            nn.Linear(100, classes))
        
    def forward(self, x, embedding=False):
        if embedding: 
            return self.embedding_model(x)

        else:
            embedding = self.embedding_model(x)
            return self.classification_model(embedding)

        
