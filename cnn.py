import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable

#Hyperparameters
num_epochs = 10
f3_no = 100
f4_no = 100
f5_no = 100
learning_rate = .001
l2_const = 3
EMBEDDING_DIM = 300

#REMOVE LATER
torch.manual_seed(7)

class CNN(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CNN, self).__init__()
        
        #Convolution layers in size 3, 4 and 5 windows
        self.conv3 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(3, embedding_size)) for f in range(f3_no)])
        self.conv4 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(4, embedding_size), padding=(1, 0)) for f in range(f4_no)])
        self.conv5 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(5, embedding_size), padding=(2, 0)) for f in range(f5_no)])
        
        #Fully connected layer 
        self.fc = nn.Linear(f3_no + f4_no + f5_no, num_classes)
        
        #.5 dropout on FC (penultimate) layer
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        conv3s = [F.relu(conv(x)).squeeze(3) for conv in self.conv3] #(batch len, feature len, feature width (1))
        conv4s = [F.relu(conv(x)).squeeze(3) for conv in self.conv4]
        conv5s = [F.relu(conv(x)).squeeze(3) for conv in self.conv5]
        
        maxpool3 = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2) for fmap in conv3s]
        maxpool4 = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2)  for fmap in conv4s]
        maxpool5 = [F.max_pool1d(fmap, fmap.size(2)).squeeze(2)  for fmap in conv5s]
        
        join_max = torch.cat((torch.cat(maxpool3, 1), torch.cat(maxpool4, 1), torch.cat(maxpool5, 1)), 1)
    
        fc_layer = self.dropout(self.fc(join_max))
        
        out = F.log_softmax(fc_layer, dim=1)
        
        return out
    

train_iter, val_iter, test_iter, TEXT, LABEL = pp.preprocess_SST()


model = CNN(2, EMBEDDING_DIM)

# Loss and Optimizer
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=3)  
losses = []

# Training the Model
for epoch in range(num_epochs):
    total_loss = torch.Tensor([0])
    for i, batch in enumerate(train_iter):
        print(i)
        print(batch.text.size())
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x = Variable(torch.from_numpy(pp.embeddings(batch, TEXT)).float().unsqueeze(1))
        y = batch.label - 1
        outputs = model(x)
    
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
    losses.append(total_loss)
print(losses)  


print("testing...")

# Test the Model
correct = 0
total = 0
for batch in test_iter:
    x = Variable(torch.from_numpy(pp.embeddings(batch, TEXT)).float().unsqueeze(1))
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)
    
    #predicted = np.where(predicted == 0, 1, 2)
    
    total += batch.text.size()[1]
    correct += (predicted == (batch.label.data[0] - 1)).sum()

print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))