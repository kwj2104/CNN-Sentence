import torch
import torch.nn as nn
import preprocessing as pp
import torch.nn.functional as F


# Hyper Parameters 
num_epochs = 10
learning_rate = 0.001
embedding_dim = 100
hl1_size = 128

dtype = torch.FloatTensor

# Model
class CBOWRegression(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(CBOWRegression, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hl1_size)
        self.linear2 = nn.Linear(hl1_size, num_classes)
    
    def forward(self, inputs):
        #Sum CBOW
        embeds = self.embeddings(context_var).sum(1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

train_iter, val_iter, test_iter, TEXT, LABEL = pp.preprocess_SST()

model = CBOWRegression(len(LABEL.vocab), len(TEXT.vocab))

# Loss and Optimizer
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )  
losses = []


# Training the Model
for epoch in range(num_epochs):
    total_loss = torch.Tensor([0])
    for i, batch in enumerate(train_iter):
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        context_var = batch.text.transpose(0, 1)
        outputs = model(context_var)
        loss = criterion(outputs, batch.label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
    losses.append(total_loss)
print(losses)  
        

# Test the Model
correct = 0
total = 0
for batch in test_iter:
    context_var = batch.text.transpose(0, 1)
    outputs = model(context_var)
    _, predicted = torch.max(outputs.data, 1)
    
    total += batch.text.size()[1]
    correct += (predicted == batch.label.data[0]).sum()

print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))

