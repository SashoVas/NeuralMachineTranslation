import time
from nltk.translate.bleu_score import corpus_bleu
import math
import nltk
import torch
from parameters import *
import utils

max_lr = 2e-3
min_lr = 1e-4
warmup_steps = 3000
decay_steps = 220_000


def get_lr(step):
    if step < warmup_steps:
        return max_lr*(step+1)/warmup_steps

    if step > decay_steps:
        return min_lr

    decay_ratio = (step - warmup_steps)/(decay_steps-warmup_steps)
    coef = 0.5*(1 + math.cos(math.pi * decay_ratio))
    return min_lr + coef*(max_lr-min_lr)


def test_performance(model, data_loader, optimizer, word2ind, iterations=100):
    results = []
    model.train()
    for i in range(iterations):
        t0 = time.time()
        X = data_loader.next_batch().to(device=device)

        logits = model(X[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.flatten(
            0, 1), X[:, 1:].flatten(0, 1), ignore_index=word2ind[padToken])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (data_loader.batch_size *
                          data_loader.context_length)/(t1-t0)
        results.append([dt, tokens_per_sec])
        print(
            f"step:{i}, loss:{loss.item()}, dt:{dt:.2f}ms, tok/sec:{tokens_per_sec:.2f}")
    return results


def test(model, data_loader, word2ind):
    model.eval()

    sum_loss = 0
    H = 0
    c = 0
    with torch.no_grad():
        for j in range(len(data_loader)):
            X = data_loader.next_batch().to(device=device)

            logits = model(X[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.flatten(
                0, 1), X[:, 1:].flatten(0, 1), ignore_index=word2ind[padToken])
            sum_loss += loss.item()

            l = sum(len(s)-1 for s in X)
            c += l
            H += l * loss.item()
    return sum_loss/len(data_loader), math.exp(H/c)


def train(model, data_loader, optimizer, word2ind, epochs=10, test_dataloader=None, log_iterations=100, clip_grad=None, smoothing=0, step=0):
    best_perplexity = float("inf")
    train_loss = []
    test_loss = []
    perplexity = []
    step = step
    for i in range(epochs):
        model.train()
        current_train_loss = []
        for j in range(len(data_loader)):
            X = data_loader.next_batch().to(device=device)

            logits = model(X[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), X[:, 1:].flatten(
                0, 1), ignore_index=word2ind[padToken], label_smoothing=smoothing)

            optimizer.zero_grad()
            loss.backward()
            if clip_grad != None:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad)

            # learning rate scheduler
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            if j % log_iterations == 0:
                print(f"Epoch:{i} Iter{j} loss:{loss.item()}")
            current_train_loss.append(loss.item())
            step += 1

        train_loss.append(current_train_loss)

        if test_dataloader is not None:
            current_test_loss, current_perplexity = test(
                model, test_dataloader, word2ind)
            test_loss.append(current_test_loss)
            perplexity.append(current_perplexity)
            print(
                f"Epoch:{i} loss:{current_test_loss}, perplexity:{current_perplexity}")

            if current_perplexity < best_perplexity:
                best_perplexity = current_perplexity
                print('Saving new best model.')
                model.save(modelFileName)
                torch.save((iter, best_perplexity, learning_rate,
                           optimizer.state_dict()), modelFileName + '.optim')
                model.save('/content/drive/MyDrive/' + modelFileName)
                torch.save((iter, best_perplexity, learning_rate, optimizer.state_dict(
                )), '/content/drive/MyDrive/' + modelFileName + '.optim')

    return train_loss, test_loss, perplexity


def test_translate(model, word2ind, space_between_words=True):
    ind2word = {j: i for i, j in word2ind.items()}
    sourceTest = utils.readCorpus('en_bg_data/test.bg')
    test = [[startToken] + s + [transToken] for s in sourceTest]
    test = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test]

    translated = []
    with open("translate_result.txt", 'w', encoding='utf-8') as file:
        for input in test:
            res = model.generate(input, mode='argmax')
            if len(res) == 0:
                translated.append(["Too big"])
                file.write("Too big"+"\n")
                continue
            result = [ind2word[i]
                      for i in (res[:-1] if res[-1] == word2ind[endToken] else res)]
            translated.append(result)
            if space_between_words:
                file.write(' '.join(result)+"\n")
            else:
                file.write(''.join(result)+"\n")

    ref = [[s] for s in utils.readCorpus("en_bg_data/test.en")]
    # hyp = utils.readCorpus("translate_result.txt")
    bleu_score = corpus_bleu(ref, translated)
    return bleu_score, translated


def test_translate_tokenized(model, tokenizer, testCorpus, word2ind, mask_bulgarian=False):
    translated = []
    ref = [[s] for s in utils.readCorpus("en_bg_data/test.en")]

    concatenated = '\n'.join([' '.join(s)
                             for s in utils.readCorpus("en_bg_data/test.en")])
    symbols = set(concatenated).union({'<', '>', '/'})
    bulgarian_tokens = [
        i for i, j in tokenizer.vocab.items() if not set(j).issubset(symbols)]
    mask = torch.zeros(tokenizer.vocab_size).to(device=device)
    mask[bulgarian_tokens] = 1
    mask = mask == 1

    with open("translate_result.txt", 'w', encoding='utf-8') as file:
        for text in testCorpus:
            if not mask_bulgarian:
                res = model.generate(text)
            else:
                res = model.generate(text, mask=mask)
            if len(res) == 0:
                translated.append("Too big")
                file.write("Too big"+"\n")
                continue
            result = tokenizer.decode(
                res[:-1] if res[-1] == word2ind[endToken] else res)
            result = ''.join(result)
            if result[-1] != "\n":
                result += "\n"
            file.write(result)
            translated.append(result)
    translated_split = [nltk.word_tokenize(i) for i in translated]

    # hyp = utils.readCorpus("translate_result.txt")
    bleu_score = corpus_bleu(ref, translated_split)
    return bleu_score, translated
