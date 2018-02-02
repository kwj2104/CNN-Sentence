import torch
import torch.nn as nn
import preprocessing as pp
from torch.autograd import Variable


# Hyper Parameters 
num_epochs = 30
learning_rate = 0.001
dtype = torch.FloatTensor



# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        
        out = self.linear(x)
        return out


train_iter, val_iter, test_iter, TEXT, LABEL = pp.preprocess_SST()

model = LogisticRegression(len(TEXT.vocab), len(LABEL.vocab))

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )  

# Training the Model
for epoch in range(num_epochs):
    for i, batch in enumerate(train_iter):
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x_oh = Variable(torch.from_numpy(pp.one_hot(batch, TEXT))).type(dtype)
        outputs = model(x_oh)
        loss = criterion(outputs, batch.label)
        loss.backward()
        optimizer.step()

# Test the Model
correct = 0
total = 0
for batch in test_iter:
    x_oh = Variable(torch.from_numpy(pp.one_hot(batch, TEXT))).float()
    outputs = model(x_oh)
    _, predicted = torch.max(outputs.data, 1)
    
    total += batch.text.size()[1]
    correct += (predicted == batch.label.data[0]).sum()

print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))

