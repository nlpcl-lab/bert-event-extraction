import os
import argparse

from sklearn.metrics import classification_report
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, trigger2idx, idx2trigger, all_arguments, tokenizer
from consts import NONE, PAD
from utils import calc_metric


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model(tokens_x_2d, entities_x_3d, head_indexes_2d, triggers_y_2d, arguments_2d)

        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d = model.module.argument_loss(argument_hidden, argument_keys, arguments_2d)
            argument_loss = criterion(argument_logits, arguments_y_1d)

            loss = trigger_loss + argument_loss
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("tokens:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
            print("entities_x_3d:", entities_x_3d[0][:seqlens_1d[0]])
            print("head_indexes_2d:", head_indexes_2d[0][:seqlens_1d[0]])
            print("triggers:", triggers_2d[0])
            print("triggers_y:", triggers_y_2d[0][:seqlens_1d[0]])
            print('triggers_y_hat:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print("arguments_2d:", arguments_2d[0])
            print("seqlen:", seqlens_1d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


def eval(model, iterator, fname):
    model.eval()

    words_all, trigger_all, trigger_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model(tokens_x_2d, entities_x_3d, head_indexes_2d, triggers_y_2d, arguments_2d)

            words_all.extend(words_2d)
            trigger_all.extend(triggers_2d)
            trigger_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())

    # save
    with open('temp', 'w') as fout:
        for words, triggers, trigger_hat in zip(words_all, trigger_all, trigger_hat_all):
            trigger_hat = trigger_hat[:len(words)]
            trigger_hat = [idx2trigger[hat] for hat in trigger_hat]

            for w, t, p in zip(words[1:-1], triggers[1:-1], trigger_hat[1:-1]):
                fout.write(f"{w}\t{t}\t{p}\n")
            fout.write("\n")

    y_true, y_pred = [], []
    with open('temp', 'r') as fout:
        lines = fout.read().splitlines()
        for line in lines:
            if len(line) > 0:
                y_true.append(trigger2idx[line.split('\t')[1]])
                y_pred.append(trigger2idx[line.split('\t')[2]])

    precision, recall, f1 = calc_metric(y_true, y_pred)
    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

    if f1 > 0.69:
        final = fname + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
        with open(final, 'w') as fout:
            result = open("temp", "r").read()
            fout.write(f"{result}\n")
    os.remove("temp")
    print('[classification]\t\tP={:.3f}\tR={:.3f}\tF1={:.3f}'.format(precision, recall, f1))

    y_true = list(map(lambda x: 2 if x >= 2 else x, y_true))
    y_pred = list(map(lambda x: 2 if x >= 2 else x, y_pred))
    precision, recall, f1 = calc_metric(y_true, y_pred)
    print('[identification]\t\tP={:.3f}\tR={:.3f}\tF1={:.3f}'.format(precision, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        entity_size=len(all_entities),
        argument_size=len(all_arguments)
    )
    if device == 'cuda':
        model = model.cuda()

    model = nn.DataParallel(model)

    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)

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
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    # weight = torch.ones([len(all_triggers)]) * 2
    # weight[trigger2idx[NONE]] = 1.0
    # weight = weight.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion)

        print(f"=========eval at epoch={epoch}=========")
        fname = os.path.join(hp.logdir, str(epoch))

        # eval(model, train_iter, fname + '_train')
        eval(model, dev_iter, fname + '_dev')
        eval(model, test_iter, fname + '_test')

        torch.save(model.state_dict(), "latest_model.pt")
        # print(f"weights were saved to {fname}.pt")
