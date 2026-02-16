import sys
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')


class progressBar:
    def __init__(self, barWidth=50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item = 0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))

    def tick(self):
        if self.item > 0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")


def readCorpus(fileName):
    print('Loading file:', fileName)
    with open(fileName, encoding='utf-8') as f:
        return [nltk.word_tokenize(line) for line in f]


def getDictionary(corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold=2):
    dictionary = {}
    for s in corpus:
        for w in s:
            if w in dictionary:
                dictionary[w] += 1
            else:
                dictionary[w] = 1

    words = [startToken, endToken, unkToken, padToken, transToken] + \
        [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    return {w: i for i, w in enumerate(words)}


def prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold=2):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    word2ind = getDictionary(sourceCorpus+targetCorpus,
                             startToken, endToken, unkToken, padToken, transToken, wordCountThreshold=wordCountThreshold)

    trainCorpus = [[startToken] + s + [transToken] + t + [endToken]
                   for (s, t) in zip(sourceCorpus, targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [[startToken] + s + [transToken] + t + [endToken]
                 for (s, t) in zip(sourceDev, targetDev)]

    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind
