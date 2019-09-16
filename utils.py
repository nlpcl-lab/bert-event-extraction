from consts import NONE, PAD


def build_vocab(labels):
    all_labels = [PAD, NONE]
    for label in labels:
        all_labels.append('B-{}'.format(label))
        all_labels.append('I-{}'.format(label))
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label
