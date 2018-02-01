import torchtext
from torchtext.vocab import Vectors, GloVe
import numpy as np


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
        (train, val, test), batch_size=10, device=-1, repeat=False)
    
    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    
    #print("Word embeddings size ", TEXT.vocab.vectors.size())
    #print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])

    return train_iter, val_iter, test_iter, TEXT, LABEL


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

def embeddings(batch, TEXT):
    batch_size = batch.text.size()[1]
    vocab_len = len(TEXT.vocab)
    batch_embed = np.zeros((batch_size, vocab_len))
    for x in range(batch_size):
        sent_bow = np.zeros(vocab_len)
        for word_index in batch.text[:, x].data:
            sent_bow += TEXT.vocab.vectors[word_index]
        batch_embed[x, :] = sent_bow
    
    return batch_embed


        
            

    
#train_iter, val_iter, test_iter, TEXT, LABEL = preprocess_SST()

#batch = next(iter(train_iter))

#test_batch_oh = one_hot(batch, TEXT)

#print(test_batch_oh.shape)
#print(sum(test_batch_oh[1,:]))
