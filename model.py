import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, trigger_size=None, entity_size=None, entity_embedding_dim=50, device='cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768 + entity_embedding_dim, hidden_size=768 // 2, batch_first=True)
        # self.fc = nn.Linear(768 + entity_embedding_dim, trigger_size)
        self.fc = nn.Sequential(
            nn.Linear(768 + entity_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, trigger_size),
        )
        self.entity_embedding = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)

        self.device = device

    def forward(self, tokens_x_2d, entities_x_3d, triggers_y_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(tokens_x_2d)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]

        entities_embedding = self.entity_embedding(entities_x_3d)
        out = torch.cat([enc, entities_embedding], 2)
        # out, _ = self.rnn(out)
        # out.shape: [batch_size, seq_len, num_directions * hidden_size]

        logits = self.fc(out)
        y_hat = logits.argmax(-1)
        return logits, triggers_y_2d, y_hat


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
