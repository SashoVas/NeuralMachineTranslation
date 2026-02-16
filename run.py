import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *
from train import train, test
from dataloader import CustomDataLoader

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


if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
    trainCorpus, devCorpus, word2ind = utils.prepareData(
        sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold=5)
    trainCorpus = [[word2ind.get(w, unkTokenIdx)
                    for w in s] for s in trainCorpus]
    devCorpus = [[word2ind.get(w, unkTokenIdx) for w in s] for s in devCorpus]
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv) > 1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (trainCorpus, devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    vocab_size = len(word2ind)
    nmt = model.LanguageModel(vocab_size,
                              embed_dim,
                              d_keys,
                              d_values,
                              word2ind,
                              endToken,
                              dropout_prob=dropout_prob,
                              context_length=context_len,
                              n_heads=n_heads,
                              transformer_layers=transformer_layers).to(device)
    optimizer = torch.optim.AdamW(
        nmt.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)

    train_data_loader = CustomDataLoader(
        trainCorpus, word2ind, padToken, batchSize=batchSize, context_length=context_len)
    val_data_loader = CustomDataLoader(
        devCorpus, word2ind, padToken, batchSize=16, context_length=context_len, shuffle=False)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter, bestPerplexity, learning_rate,
         osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    train(nmt,
          train_data_loader,
          optimizer,
          word2ind,
          epochs=maxEpochs,
          test_dataloader=val_data_loader,
          clip_grad=clip_grad,
          smoothing=smoothing,
          step=0)

    print('reached maximum number of epochs!')
    nmt.eval()
    cross_entropy, currentPerplexity = test(nmt, val_data_loader, word2ind)
    print('Last model perplexity: ', currentPerplexity)

    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter, bestPerplexity, learning_rate,
                   optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv) > 3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    vocab_size = len(word2ind)
    nmt = model.LanguageModel(vocab_size,
                              embed_dim,
                              d_keys,
                              d_values,
                              word2ind,
                              endToken,
                              dropout_prob=dropout_prob,
                              context_length=context_len,
                              n_heads=n_heads,
                              transformer_layers=transformer_layers).to(device)
    nmt.load(modelFileName)

    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    test_data = [[startToken] + s + [transToken] + t + [endToken]
                 for (s, t) in zip(sourceTest, targetTest)]
    test_data = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test_data]

    val_data_loader = CustomDataLoader(
        test_data, word2ind, padToken, batchSize=16, context_length=context_len, shuffle=False)
    nmt.eval()
    cross_entropy, perplexity = test(nmt, val_data_loader, word2ind)
    print('Model perplexity: ', perplexity)

if len(sys.argv) > 3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    sourceTest = utils.readCorpus(sys.argv[2])
    test_data = [[startToken] + s + [transToken] for s in sourceTest]
    test_data = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test_data]

    vocab_size = len(word2ind)
    nmt = model.LanguageModel(vocab_size,
                              embed_dim,
                              d_keys,
                              d_values,
                              word2ind,
                              endToken,
                              dropout_prob=dropout_prob,
                              context_length=context_len,
                              n_heads=n_heads,
                              transformer_layers=transformer_layers).to(device)
    nmt.load(modelFileName)
    ind2word = {j: i for i, j in word2ind.items()}
    nmt.eval()
    pb = utils.progressBar()
    pb.start(len(test_data))
    with open(sys.argv[3], 'w', encoding='utf-8') as file:
        for input in test_data:
            res = nmt.generate(input, mode='argmax')
            if len(res) == 0:
                file.write("Too big"+"\n")
                continue
            result = [ind2word[i]
                      for i in (res[:-1] if res[-1] == word2ind[endToken] else res)]
            file.write(' '.join(result)+"\n")
            pb.tick()
    pb.stop()

if len(sys.argv) > 2 and sys.argv[1] == 'generate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    test = sys.argv[2].split()
    test = [word2ind.get(w, unkTokenIdx) for w in test]

    vocab_size = len(word2ind)
    nmt = model.LanguageModel(vocab_size,
                              embed_dim,
                              d_keys,
                              d_values,
                              word2ind,
                              endToken,
                              dropout_prob=dropout_prob,
                              context_length=context_len,
                              n_heads=n_heads,
                              transformer_layers=transformer_layers).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    r = nmt.generate(test)
    result = [words[i] for i in r]
    print(' '.join(result)+"\n")

if len(sys.argv) > 3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
