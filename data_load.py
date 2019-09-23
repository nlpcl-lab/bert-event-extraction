import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES
from utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.triggers_li, self.arguments_li, self.entities_li = [], [], [], []

        cut_off = 50
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words'][:cut_off]
                triggers = [NONE] * len(words)
                arguments = [NONE] * len(words)
                entities = [[NONE] for _ in range(len(words))]

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        if i >= cut_off:
                            continue
                        event_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(event_type)
                        else:
                            triggers[i] = 'I-{}'.format(event_type)

                    for argument in event_mention['arguments']:
                        argument_role = argument['role']
                        for j in range(argument['start'], argument['end']):
                            if j >= cut_off:
                                continue
                            if j == argument['start']:
                                arguments[j] = 'B-{}'.format(argument_role)
                            else:
                                arguments[j] = 'I-{}'.format(argument_role)

                for entity_mention in item['golden-entity-mentions']:
                    for i in range(entity_mention['start'], entity_mention['end']):
                        if i >= cut_off:
                            continue
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                self.sent_li.append([CLS] + words + [SEP])
                self.triggers_li.append([PAD] + triggers + [PAD])
                self.arguments_li.append([PAD] + arguments + [PAD])
                self.entities_li.append([[PAD]] + entities + [[PAD]])

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, triggers, arguments, entities = self.sent_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.entities_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x = [], []
        triggers_y, arguments_y = [], []
        is_heads = []
        for w, t, a, e in zip(words, triggers, arguments, entities):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + [PAD] * (len(tokens) - 1)  # <PAD>: no decision
            a = [a] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)

            t = [trigger2idx[each] for each in t]
            a = [argument2idx[each] for each in a]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            is_heads.extend(is_head)
            tokens_x.extend(tokens_xx), entities_x.extend(e)
            triggers_y.extend(t), arguments_y.extend(a)

        assert len(tokens_x) == len(entities_x) == len(triggers_y) == len(arguments_y), \
            'len(tokens_x)={}, len(entities_x)={}, len(triggers_y)={}, len(triggers_y)={}'.format(len(tokens_x), len(entities_x), len(arguments_y), len(arguments_y))

        seqlen = len(triggers_y)

        return tokens_x, entities_x, triggers_y, arguments_y, seqlen, is_heads, words, triggers


def pad(batch):
    tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_y_2d, seqlens_1d, is_heads_2d, words_2d, triggers_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        arguments_y_2d[i] = arguments_y_2d[i] + [argument2idx[PAD]] * (maxlen - len(arguments_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]

    return tokens_x_2d, entities_x_3d, \
           triggers_y_2d, arguments_y_2d, \
           seqlens_1d, is_heads_2d, words_2d, triggers_2d
