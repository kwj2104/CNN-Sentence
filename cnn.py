import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

f3_no = 1
f4_no = 1
f5_no = 1


class CNN(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CNN, self).__init__()
        
        #Convolution layers in size 3, 4 and 5 windows
        self.conv3 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(3, embedding_size)) for f in f3_no])
        self.conv4 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(4, embedding_size)) for f in f4_no])
        self.conv5 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(5, embedding_size)) for f in f5_no])
        
        #Fully connected layer 
        self.fc = nn.Linear(len(f3_no) + len(f4_no) + len(f5_no), num_classes)
        
        #.5 dropout on FC (penultimate) layer
        self.dropout = nn.dropout()
        
        
        
    def forward(self, x):
        conv3s = [F.relu(conv(x)) for conv in self.conv3]
        conv4s = [F.relu(conv(x)) for conv in self.conv4]
        conv5s = [F.relu(conv(x)) for conv in self.conv5]
        
        maxpool3 = [F.max_pool1d(fmap, fmap.size(2)) for fmap in conv3s]
        maxpool4 = [F.max_pool1d(fmap, fmap.size(2)) for fmap in conv4s]
        maxpool5 = [F.max_pool1d(fmap, fmap.size(2)) for fmap in conv5s]
        
        join_max = torch.cat((maxpool3, maxpool4, maxpool5), 1)
        
        fc_layer = self.dropout(self.fc(join_max))
        
        out = F.log_softmax(fc_layer)
        
        return out
    
    

train_iter, val_iter, test_iter, TEXT, LABEL = pp.preprocess_SST()

batch = next(iter(train_iter))

x = torch.from_numpy(pp.embeddings(batch, TEXT)).float()