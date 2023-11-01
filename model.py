import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x * math.sqrt(self.model_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, seq_len: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, model_dim)
        pe = torch.zeros(seq_len, model_dim)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension to pe matrix with a shape (1, seq_len, model_dim)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, 0 : x.shape[1], :]).requires_grad(False)
        x = self.dropout(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** (-6)):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x has a shape (batch, seq_len, model_dim)
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_num: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        assert model_dim % head_num == 0, "model_dim is not divisible by head_num!"

        self.submodel_dim = model_dim // head_num
        self.query_weight = nn.Linear(model_dim, model_dim)
        self.key_weight = nn.Linear(model_dim, model_dim)
        self.value_weight = nn.Linear(model_dim, model_dim)
        self.output_weight = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout = None):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, head_num, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores * value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.query_weight(q)  # (batch, seq_len, model_dim)
        key = self.key_weight(k)  # (batch, seq_len, model_dim)
        value = self.value_weight(v)  # (batch, seq_len, model_dim)

        # (batch, seq_len, model_dim) -> (batch, head_num, seq_len, submodel_dim)
        query = query.view(query.shape[0], query.shape[1], self.head_num, self.submodel_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head_num, self.submodel_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head_num, self.submodel_dim).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.head_num * self.submodel_dim)
        )  # (batch, head_num, seq_len, model_dim)

        x = self.output_weight(x)

        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))  # self.norm(sublayer(x)) in the original paper
