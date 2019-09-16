import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, trigger2idx, idx2trigger, tokenizer


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_y_2d, seqlens_1d, is_heads_2d, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        logits, y, _ = model(tokens_x_2d, triggers_y_2d)

        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("tokens:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
            print("is_heads:", is_heads_2d[0])
            print("triggers:", triggers_y_2d[0][:seqlens_1d[0]])
            print("seqlen:", seqlens_1d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


def eval(model, iterator, f, identification=False):
    model.eval()

    Words, Is_heads, Triggers, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_y_2d, seqlens_1d, is_heads_2d, words_2d, triggers_2d = batch

            _, _, y_hat = model(tokens_x_2d, triggers_y_2d)

            Words.extend(words_2d)
            Is_heads.extend(is_heads_2d)
            Triggers.extend(triggers_2d)
            Y.extend(triggers_y_2d)
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, triggers, y_hat in zip(Words, Is_heads, Triggers, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2trigger[hat] for hat in y_hat]

            assert len(preds) == len(words.split()) == len(triggers.split())

            for w, t, p in zip(words.split()[1:-1], triggers.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array([trigger2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([trigger2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    if identification:
        print('identification mode!')
        y_true = list(map(lambda x: 2 if x >= 2 else x, y_true))
        y_pred = list(map(lambda x: 2 if x >= 2 else x, y_pred))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print('num_proposed: {}, num_correct: {}, num_gold: {}'.format(num_proposed, num_correct, num_gold))
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.3f" % precision)
    print("recall=%.3f" % recall)
    print("f1=%.3f" % f1)
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(device=device, trigger_size=len(all_triggers)).cuda()
    model = nn.DataParallel(model)

    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = eval(model, dev_iter, fname)
        precision, recall, f1 = eval(model, dev_iter, fname, identification=True)

        torch.save(model.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")
