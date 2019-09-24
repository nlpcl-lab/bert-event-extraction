import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES
from utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.entities_li, self.triggers_li, self.arguments_li, = [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                triggers = [NONE] * len(words)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ..
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_str"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:
                    arguments['events'][event_mention['event_type']] = []

                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    for argument in event_mention['arguments']:
                        arguments['events'][event_mention['event_type']].append((argument['start'], argument['end'], argument['role']))

                self.sent_li.append([CLS] + words + [SEP])
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.triggers_li.append([PAD] + triggers + [PAD])
                self.arguments_li.append(arguments)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, triggers, arguments, = self.sent_li[idx], self.entities_li[idx], self.triggers_li[idx], self.arguments_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x = [], []
        triggers_y = []
        is_heads = []
        for w, e, t in zip(words, entities, triggers):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            # t = [t] + [PAD] * (len(tokens) - 1)  # <PAD>: no decision
            e = [e] + [[PAD]] * (len(tokens) - 1)

            t = [trigger2idx[t]]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            is_heads.extend(is_head)
            tokens_x.extend(tokens_xx)
            entities_x.extend(e)
            triggers_y.extend(t)

        for event in arguments['events']:
            for i in range(len(arguments['events'][event])):
                argument = arguments['events'][event][i]
                arguments['events'][event][i] = (argument[0], argument[1], argument2idx[argument[2]])

        head_indexes = [0]
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        # assert len(tokens_x) == len(entities_x) == len(triggers_y), \
        #     'len(tokens_x)={}, len(entities_x)={}, len(triggers_y)={}'.format(len(tokens_x), len(entities_x), len(triggers_y))

        seqlen = len(tokens_x)

        return tokens_x, entities_x, triggers_y, arguments, seqlen, head_indexes, words, triggers


def pad(batch):
    tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]

    return tokens_x_2d, entities_x_3d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d
