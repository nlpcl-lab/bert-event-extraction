import numpy as np

from consts import NONE, PAD


def build_vocab(labels):
    all_labels = [PAD, NONE]
    for label in labels:
        all_labels.append('B-{}'.format(label))
        all_labels.append('I-{}'.format(label))
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def calc_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print('proposed: {}, correct: {}, gold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1
