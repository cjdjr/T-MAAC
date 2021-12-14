import torch.nn as nn
import torch
import numpy as np

class transition_model(nn.Module):
    def __init__(self, input_dim = 1364+38, output_dim=322):
        super(transition_model, self).__init__()
        lys=[]
        self.stem=nn.Linear(input_dim , 512)
        lys.append(nn.ReLU())
        for x in range(2):
            lys.append(nn.Linear(512,512))
            # lys.append(nn.Dropout(0.5))
            lys.append(nn.ReLU())
        self.ly=nn.Sequential(*lys)
        self.out=nn.Sequential(nn.Linear(512,output_dim))

    def forward(self, x):
        x=self.stem(x)
        return self.out(self.ly(x))

class transition_model_linear(nn.Module):
    def __init__(self, input_dim = 1364+38, action_dim=38, output_dim=322):
        super(transition_model_linear, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        lys=[]
        self.stem=nn.Linear(input_dim , 1024)
        lys.append(nn.ReLU())
        for x in range(1):
            lys.append(nn.Linear(1024,1024))
            # lys.append(nn.Dropout())
            lys.append(nn.ReLU())
        lys.append(nn.Linear(1024,512))
        lys.append(nn.ReLU())
        for x in range(1):
            lys.append(nn.Linear(512,512))
            # lys.append(nn.Dropout())
            lys.append(nn.ReLU())
        self.ly=nn.Sequential(*lys)
        self.out=nn.Sequential(nn.Linear(512,output_dim*action_dim + output_dim))

    def forward(self, input):
        x=self.stem(input)
        out = self.out(self.ly(x))
        B, _ = input.shape
        q = input[:,-self.action_dim:,None]
        weight, bias = out[:,:self.output_dim * self.action_dim].view(B,self.output_dim,self.action_dim), out[:,self.output_dim * self.action_dim:]
        return torch.bmm(weight, q).squeeze(-1) + bias

    def get_coff(self, state, q):
        x = torch.cat((state,q),dim=1)
        x=self.stem(x)
        out = self.out(self.ly(x))
        B, _ = x.shape
        weight, bias = out[:,:self.output_dim * self.action_dim].view(B,self.output_dim,self.action_dim), out[:,self.output_dim * self.action_dim:]
        return weight.squeeze(dim=0).detach().cpu().numpy(), bias.squeeze(dim=0).detach().cpu().numpy()