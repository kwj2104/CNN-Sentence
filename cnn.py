import torch
import torchtext
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse as argparse

"""
Module for CNN-multichannel applied to the SST2 Dataset

This implements the model using hyperparameters described 
by Yoon Kim (2014)
http://www.aclweb.org/anthology/D14-1181
"""


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--devid", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--channelno", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bsize", type=int, default=64)
    
    return parser.parse_args()

args = parse_args()


"""
Preprocess data from SST2 Dataset
Use prebuilt word vectors from GloVe
Wikipedia 2014 + Gigaword 5 (300d vectors)
https://nlp.stanford.edu/projects/glove/
"""
def preprocess_SST():

    TEXT = torchtext.data.Field()
    LABEL = torchtext.data.Field(sequential=False)
    
    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')
    
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=args.bsize, device=args.devid, repeat=False)
    
    TEXT.vocab.load_vectors(vectors=GloVe(name='6B'))
    
    return train_iter, val_iter, test_iter, TEXT, LABEL


"""
CNN Model using both static and nonstatic word embedding convolutions
"""
class CNN(nn.Module):
    def __init__(self, embeds, dropout=.5):
        super(CNN, self).__init__()
        
        self.vocab_size = embeds.size(0)
        self.embeds_size = embeds.size(1)
        self.dropout = dropout
        
        self.embeddings = nn.Embedding(self.vocab_size, self.embeds_size)
        self.embeddings_static = nn.Embedding(self.vocab_size, self.embeds_size)
        self.embeddings.weight = nn.Parameter(embeds)
        self.embeddings_static.weight = nn.Parameter(embeds, requires_grad=False)
        
        #Convolution layers in size 3, 4 and 5 windows
        self.conv_list = nn.ModuleList([nn.Conv2d(1, args.channelno, kernel_size=(3 + i, self.embeds_size), padding=(i, 0)) for i in range(3)])
        self.conv_list_s = nn.ModuleList([nn.Conv2d(1, args.channelno, kernel_size=(3 + i, self.embeds_size), padding=(i, 0)) for i in range(3)])
        
        #Fully connected layer 
        self.fc = nn.Linear(2 * (3 * args.channelno), 2)
        
        #dropout on FC (penultimate) layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        x = self.embeddings(x.transpose(0,1)).unsqueeze(1)
        
        conv_nonstatic = [F.relu(conv(x)).squeeze(3) for conv in self.conv_list]
        conv_static = [F.relu(conv(x)).squeeze(3) for conv in self.conv_list_s]
        
        maxpool = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2) for fmap in conv_nonstatic]
        maxpool_s = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2) for fmap in conv_static]
        
        join_max = torch.cat((torch.cat(maxpool, 1), torch.cat(maxpool_s, 1)), 1)
    
        out = self.fc(self.dropout(join_max))
        
        return out


def train(train_iter, model, criterion, optimizer):
    model.train()
    total_loss = 0
    total = 0
    for batch in tqdm(train_iter):
        optimizer.zero_grad()
        x = batch.text
        y = batch.label - 1
        outputs = model(x)
    
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data[0]
        total += batch.text.size()[1]
    return total_loss / total
    

def val(val_iter, model, criterion):
    model.eval()
    total_loss = 0
    total = 0
    for batch in tqdm(val_iter):
        x = batch.text
        y = batch.label - 1
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.data[0]
        total += batch.text.size()[1]
    return total_loss / total


if __name__ == "__main__":

    train_iter, val_iter, test_iter, TEXT, LABEL = preprocess_SST()

    model = CNN(TEXT.vocab.vectors, dropout=args.dropout)
    if args.devid >= 0:
        model.cuda(args.devid)
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Training the Model
    print("Training...")
    for epoch in range(args.epoch):
        train_loss = train(train_iter, model, criterion, optimizer)
        val_loss = val(val_iter, model, criterion)
        print("Epoch: {} Train Loss: {} Val Loss: {}".format(epoch, train_loss, val_loss))
    
    #Save Model
    torch.save(model.state_dict, "CNN_best.pt")

    #Test Model
    correct = 0
    total = 0
    for batch in test_iter:
        model.eval()
        x = batch.text
        y = batch.label - 1
        outputs = model(x)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += batch.text.size()[1]
        correct += (predicted == y.data).int().sum()
    
    print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))