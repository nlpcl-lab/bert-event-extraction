import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, trigger_size=None, device='cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.fc = nn.Linear(768, trigger_size)

        self.device = device

    def forward(self, x, y, ):
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
