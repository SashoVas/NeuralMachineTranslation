import regex as re
import pickle
import numpy as np
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'
transToken = '<TRANS>'


class Tokenizer:
    def __init__(self, vocab_size, initial_symbols=None):
        self.vocab_size = vocab_size
        self.merges = {}
        self.current_index = 0
        self.compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)
        self.vocab = {}
        self.symbol_to_id = {}
        self.initial_symbols = initial_symbols

    @staticmethod
    def get_stats(dataset, stats):
        for i, j in zip(dataset, dataset[1:]):
            stats[(i, j)] = stats.get((i, j), 0) + 1

        return stats

    @staticmethod
    def update_dataset(dataset, pair, pair_key):
        result = []
        i = 1
        while i < len(dataset):
            if pair == (dataset[i-1], dataset[i]):
                result.append(pair_key)
                i += 2
                if i == len(dataset):
                    result.append(dataset[i-1])
            else:
                result.append(dataset[i-1])
                i += 1
                if i == len(dataset):
                    result.append(dataset[i-1])

        return result

    def train(self, dataset):

        if not self.initial_symbols:
            self.initial_symbols = sorted(list(set(dataset)))

        self.vocab = {i: self.initial_symbols[i]
                      for i in range(len(self.initial_symbols))}
        self.symbol_to_id = {j: i for i, j in self.vocab.items()}

        # split the dataset on chinks based on the pattern
        text_chunks = re.findall(self.compiled_pattern, dataset)

        dataset = [[self.symbol_to_id[i] for i in chunk]
                   for chunk in text_chunks]
        self.merges = {}
        self.current_index = len(self.vocab)

        while self.current_index < self.vocab_size:
            if self.current_index % 10 == 0:
                print(f"Training: {self.current_index}/{self.vocab_size}")
            stats = {}

            # calculate the pairs statistics
            for chunk in dataset:
                self.get_stats(chunk, stats)

            # get best pair and update the state
            pair = max(stats, key=lambda k: stats[k])
            self.vocab[self.current_index] = self.vocab[pair[0]] + \
                self.vocab[pair[1]]
            self.merges[pair] = self.current_index

            dataset = [self.update_dataset(
                chunk, pair, self.current_index) for chunk in dataset]

            self.current_index += 1

    def continue_training(self, dataset, new_vocab_size):
        self.vocab_size = new_vocab_size
        text_chunks = re.findall(self.compiled_pattern, dataset)
        dataset = [self.encode(chunk)
                   for chunk in text_chunks]

        while self.current_index < self.vocab_size:
            if self.current_index % 10 == 0:
                print(f"Training: {self.current_index}/{self.vocab_size}")
            stats = {}

            # calculate the pairs statistics
            for chunk in dataset:
                self.get_stats(chunk, stats)

            # get best pair and update the state
            pair = max(stats, key=lambda k: stats[k])
            self.vocab[self.current_index] = self.vocab[pair[0]] + \
                self.vocab[pair[1]]
            self.merges[pair] = self.current_index

            dataset = [self.update_dataset(
                chunk, pair, self.current_index) for chunk in dataset]

            self.current_index += 1

    def decode(self, dataset):
        return "".join([self.vocab[i] for i in dataset])

    def encode(self, dataset):
        dataset = [self.symbol_to_id[i] for i in dataset]
        while len(dataset) > 1:

            stats = {}
            self.get_stats(dataset, stats)

            pair = min(stats, key=lambda k: self.merges.get(k, float('inf')))
            if pair not in self.merges:
                break
            dataset = self.update_dataset(dataset, pair, self.merges[pair])

        return dataset

    def extend(self, to_extend, bool_extend_initial_symbols=False):
        for i in to_extend:
            self.vocab[self.current_index] = i
            if bool_extend_initial_symbols:
                self.initial_symbols.append(i)
                self.symbol_to_id = {j: i for i, j in self.vocab.items()}
                self.symbol_to_id[i] = self.current_index
            self.current_index += 1

        self.vocab_size = self.current_index

    def merge(self, other):
        for key, val in other.vocab.items():
            if val in self.initial_symbols:
                continue
            self.vocab[key+self.current_index] = val

        for symbol, id in other.symbol_to_id.items():
            if symbol in self.initial_symbols:
                continue
            self.symbol_to_id[symbol] = id+self.current_index

        for i in other.initial_symbols:
            if val in self.initial_symbols:
                continue
            self.initial_symbols.append(i)

        for pair, val in other.merges.items():
            first, second = pair

            self.merges[(first + self.current_index, second +
                         self.current_index)] = self.current_index+val

        self.vocab_size = len(self.vocab) + len(self.initial_symbols)
        self.current_index = max(self.vocab.keys()) + 1

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer


def tokenize_corpus(tokenizer, suffix):
    files = ['train', 'dev', 'test']
    for i in files:
        with open(f"en_bg_data/{i}.en", 'r', encoding='utf-8') as f:
            res = f.readlines()
        res = [tokenizer.encode(i) for i in res]

        with open(f"tokenized_data/{i}_{suffix}.en", 'wb') as f:
            pickle.dump(res, f)

        with open(f"en_bg_data/{i}.bg", 'r', encoding='utf-8') as f:
            res = f.readlines()
        res = [tokenizer.encode(i) for i in res]

        with open(f"tokenized_data/{i}_{suffix}.bg", 'wb') as f:
            pickle.dump(res, f)


def prepare_data(tokenizer, suffix):
    with open(f"tokenized_data/train_{suffix}.en", 'rb') as f:
        train_en = pickle.load(f)

    with open(f"tokenized_data/train_{suffix}.bg", 'rb') as f:
        train_bg = pickle.load(f)

    train = [
        [tokenizer.symbol_to_id[startToken]] + train_bg[i] +
        [tokenizer.symbol_to_id[transToken]] +
        train_en[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(len(train_en))]

    with open(f"tokenized_data/dev_{suffix}.en", 'rb') as f:
        dev_en = pickle.load(f)

    with open(f"tokenized_data/dev_{suffix}.bg", 'rb') as f:
        dev_bg = pickle.load(f)

    dev = [
        [tokenizer.symbol_to_id[startToken]] + dev_bg[i] +
        [tokenizer.symbol_to_id[transToken]] +
        dev_en[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(len(dev_en))]

    with open(f"tokenized_data/test_{suffix}.bg", 'rb') as f:
        test_bg = pickle.load(f)

    test = [
        [tokenizer.symbol_to_id[startToken]] +
        test_bg[i] + [tokenizer.symbol_to_id[transToken]]
        for i in range(len(test_bg))]
    return train, dev, test


def prepare_data_with_pretrained(tokenizer, suffix):
    with open(f"tokenized_data/train_{suffix}.en", 'rb') as f:
        train_en = pickle.load(f)

    with open(f"tokenized_data/train_{suffix}.bg", 'rb') as f:
        train_bg = pickle.load(f)

    middle = len(train_en)//2
    pre_train_first = [
        [tokenizer.symbol_to_id[startToken]] + train_bg[i] +
        [tokenizer.symbol_to_id[transToken]] +
        train_en[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(middle)]
    pre_train_second = [
        [tokenizer.symbol_to_id[startToken]] + train_en[i] +
        [tokenizer.symbol_to_id[transToken]] +
        train_bg[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(middle)]

    pre_train = pre_train_first + pre_train_second
    np.random.shuffle(pre_train)

    train = [
        [tokenizer.symbol_to_id[startToken]] + train_bg[i] +
        [tokenizer.symbol_to_id[transToken]] +
        train_en[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(len(train_en))]

    with open(f"tokenized_data/dev_{suffix}.en", 'rb') as f:
        dev_en = pickle.load(f)

    with open(f"tokenized_data/dev_{suffix}.bg", 'rb') as f:
        dev_bg = pickle.load(f)

    dev = [
        [tokenizer.symbol_to_id[startToken]] + dev_bg[i] +
        [tokenizer.symbol_to_id[transToken]] +
        dev_en[i] + [tokenizer.symbol_to_id[endToken]]
        for i in range(len(dev_en))]

    with open(f"tokenized_data/test_{suffix}.bg", 'rb') as f:
        test_bg = pickle.load(f)

    test = [
        [tokenizer.symbol_to_id[startToken]] +
        test_bg[i] + [tokenizer.symbol_to_id[transToken]]
        for i in range(len(test_bg))]
    return pre_train, train, dev, test
