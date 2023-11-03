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
        # Apply embedding layer followed by scaling
        x = self.embedding(x)
        x = x * math.sqrt(self.model_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, seq_len: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding for input sequences
        pe = torch.zeros(seq_len, model_dim)  # (seq_len, model_dim)
        # Create a vector of shape
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add positional encoding and apply dropout
        pe = pe.unsqueeze(0)  # (1, seq_len, model_dim)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding and apply dropout
        x = x + (self.pe[:, 0 : x.shape[1], :]).requires_grad(False)
        x = self.dropout(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Apply layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply feed-forward transformation with ReLU activation and dropout
        x = torch.relu(self.linear_1(x))  # (batch, seq_len, model_dim)
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
        output = torch.matmul(attention_scores, value)

        return output, attention_scores

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
        # Apply residual connection with layer normalization and dropout
        x = x + self.dropout(sublayer(self.norm(x)))  # self.norm(sublayer(x)) in the original paper
        return x


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        self_attention_output = self.residual_connection[0](x, self.self_attention(x, x, x, src_mask))

        feed_forward_output = self.residual_connection[1](
            self_attention_output, self.feed_forward(self_attention_output)
        )

        return feed_forward_output


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attention_output = self.residual_connection[0](x, self.self_attention(x, x, x, tgt_mask))

        cross_attention_output = self.residual_connection[1](
            self_attention_output, self.cross_attention(self_attention_output, encoder_output, encoder_output, src_mask)
        )

        feed_forward_output = self.residual_connection[2](
            cross_attention_output, self.feed_forward(cross_attention_output)
        )

        return feed_forward_output


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Projectionlayer(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        x = torch.log_softmax(x, dime=-1)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: Projectionlayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    def project(self, x):
        x = self.projection_layer(x)
        return x


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    model_dim: int = 512,
    Block_num: int = 6,
    head_num: int = 8,
    dropout: float = 0.1,
    ff_dim: int = 2048,
):
    # Create the embedding layers
    src_embed = InputEmbeddings(model_dim, src_vocab_size)
    tgt_embed = InputEmbeddings(model_dim, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(model_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(model_dim, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(Block_num):
        encoder_self_attention_block = MultiHeadAttention(model_dim, head_num, dropout)
        feed_forward_block = FeedForward(model_dim, ff_dim, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(Block_num):
        decoder_self_attention_block = MultiHeadAttention(model_dim, head_num, dropout)
        decoder_cross_attention_block = MultiHeadAttention(model_dim, head_num, dropout)
        feed_forward_block = FeedForward(model_dim, ff_dim, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = Projectionlayer(model_dim, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
