import torchtext
from torchtext.vocab import Vectors, GloVe
import numpy as np
import torch as torch


def preprocess_SST():
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)
    
    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')
    

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    #print('len(TEXT.vocab)', len(TEXT.vocab))
    #print('len(LABEL.vocab)', len(LABEL.vocab))
    
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=50, device=-1, repeat=False)
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
    #print("Word embeddings size ", TEXT.vocab.vectors.size())
    #print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']])

    return train_iter, val_iter, test_iter, TEXT, LABEL

#one hot vectors
def one_hot(batch, TEXT):
    batch_size = batch.text.size()[1]
    vocab_len = len(TEXT.vocab)
    batch_onehot = np.zeros((batch_size, vocab_len))
    for x in range(batch_size):
        sent_bow = np.zeros(vocab_len)
        for word_index in batch.text[:, x].data:
            sent_bow[word_index] += 1
        batch_onehot[x, :] = sent_bow
    
    return batch_onehot


#prebuilt embeddings
def embeddings(batch, TEXT):
    batch_size = batch.text.size()[1]
    #print(batch_size)
    sent_len = batch.text.size()[0]
    #print(sent_len)
    #print(TEXT.vocab.vectors[batch.text[0, 0].data])
    embed_dim = 300
    #vocab_len = len(TEXT.vocab)
    batch_embed = np.zeros((batch_size, sent_len, embed_dim))
    for x in range(batch_size):
        sent_embed = np.array([TEXT.vocab.vectors[word_index].numpy() for word_index in batch.text[:, x].data])
        #print(sent_embed.shape)
        
        batch_embed[x, :, :] = sent_embed
        #print(batch_embed.shape)
    
    return batch_embed


        
            

    
#train_iter, val_iter, test_iter, TEXT, LABEL = preprocess_SST()

#batch = next(iter(train_iter))
#print(batch.text.transpose(0, 1))
    
#print(word_to_ix)

#batch = next(iter(train_iter))

#test_batch_oh = one_hot(batch, TEXT)

#print(test_batch_oh.shape)
#print(sum(test_batch_oh[1,:]))
