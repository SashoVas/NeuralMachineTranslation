import torch

startToken = '<S>'
startTokenIdx = 0
endToken = '</S>'
endTokenIdx = 1
unkToken = '<UNK>'
unkTokenIdx = 2
padToken = '<PAD>'
padTokenIdx = 3
transToken = '<TRANS>'
transTokenIdx = 4

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
# device = torch.device("cpu")

embed_dim = 256
context_len = 256
n_heads = 8
d_keys = 256//n_heads
d_values = 256//n_heads
transformer_layers = 8
smoothing = 0
weight_decay = 0.01
dropout_prob = 0.1

learning_rate = 1e-3
batchSize = 16
clip_grad = 3.0

maxEpochs = 10
log_every = 10
test_every = 2000
