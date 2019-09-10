import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, TRIGGERS, ARGUMENTS, ENTITIES
from utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.triggers_li, self.arguments_li, self.entities_li = [], [], [], []
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']
                triggers = [NONE] * len(words)
                arguments = [NONE] * len(words)
                entities = [[]] * len(words)

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        event_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(event_type)
                        else:
                            triggers[i] = 'I-{}'.format(event_type)

                    for argument in event_mention['arguments']:
                        argument_role = argument['role']
                        for j in range(argument['start'], argument['end']):
                            if j == argument['start']:
                                arguments[j] = 'B-{}'.format(argument_role)
                            else:
                                arguments[j] = 'I-{}'.format(argument_role)

                for entity_mention in item['golden-entity-mentions']:
                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entities[i].append('B-{}'.format(entity_type))
                        else:
                            entities[i].append('I-{}'.format(entity_type))

                self.sent_li.append([CLS] + words + [SEP])
                self.triggers_li.append([PAD] + triggers + [PAD])
                self.arguments_li.append([PAD] + arguments + [PAD])
                self.entities_li.append([PAD] + entities + [PAD])

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, triggers, arguments, entities = self.sent_li[idx], self.triggers_li, self.arguments_li, self.entities_li

        # We give credits only to the first piece.
        tokens_x, entities_x = [], []
        triggers_y, arguments_y = [], []
        is_heads = []
        for w, t, a, e in zip(words, triggers, arguments, entities):
            tokens = tokenizer.tokenize(w) if w not in (CLS, SEP) else [w]
            tokens = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + [PAD] * (len(tokens) - 1)  # <PAD>: no decision
            a = [a] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)

            t = [trigger2idx[each] for each in t]
            a = [argument2idx[each] for each in a]
            e = [[entity2idx[entity] for entity in each] for each in e]

            is_heads.extend(is_head)
            tokens_x.extend(t), entities_x.extend(e)
            triggers_y.extend(t), arguments_y.extend(a)

        seqlen = len(triggers_y)

        return tokens_x, entities_x, triggers_y, arguments_y, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <s>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
