import torch
import numpy as np


class CustomDataLoader:
    def __init__(self, trainCorpus, word2ind, padToken, batchSize=32, context_length=128, shuffle=True):
        self.trainCorpus = trainCorpus
        self.word2ind = word2ind
        self.current_idx = 0
        self.batch_size = batchSize
        self.context_length = context_length
        self.pad_token = padToken
        self.shuffle = shuffle

    def __len__(self):
        return len(self.trainCorpus)//self.batch_size

    def next_batch(self):
        initual_batch = self.trainCorpus[self.current_idx:self.current_idx+self.batch_size]
        max_len = min(self.context_length + 1, max(len(i)
                      for i in initual_batch))
        batch = torch.tensor([
            i + [self.word2ind[self.pad_token]]*(max_len-len(i))
            if len(i) < max_len else
            i[:max_len]
            for i in initual_batch
        ])
        self.current_idx += self.batch_size
        if self.current_idx+self.batch_size > len(self.trainCorpus):
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.trainCorpus)
        return batch
