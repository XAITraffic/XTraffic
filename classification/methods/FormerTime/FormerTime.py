import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model, grad=True):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)
        self.grad = grad

    def forward(self, x):
        batch_size = x.size(0)
        if not self.grad:
            with torch.no_grad():
                return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # self.attn = p_attn

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, tr=2, data_len=5000):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.attention = Attention()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.tr = tr
        self.scale = self.d_k ** -0.5
        if tr > 1:
            self.tr_layer = nn.Conv1d(data_len, data_len // tr, 1)
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.h, C // self.h).permute(0, 2, 1, 3)
        if self.tr > 1:
            x_ = self.norm(self.tr_layer(x))
            k = self.k(x_).reshape(B, -1, self.h, C // self.h).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.h, C // self.h).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.h, C // self.h).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.h, C // self.h).permute(0, 2, 1, 3)
        x, attn = self.attention(q, k, v, mask=None, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))  # layer_norm
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))  # layer_norm


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, tr, data_len, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout, tr, data_len)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x))
        x = self.skipconnect2(x, self.ffn)
        return x


class Encoder(nn.Module):
    """
    encoder in FormerTime
    """

    def __init__(self, slice_size, data_shape, d_encoder, attn_heads, enable_res_parameter, device, tr,
                 stride, layers, position_location, position_type):
        super(Encoder, self).__init__()
        self.stride = (stride, data_shape[1])
        self.slice_size = slice_size
        self.data_shape = data_shape
        self.device = device
        self.max_len = self.data_shape[0]
        self.position_location = position_location
        self.position_type = position_type

        self.input_projection = nn.Conv1d(self.slice_size[1], d_encoder, kernel_size=self.slice_size[0],
                                          stride=self.stride[0])
        self.input_norm = nn.LayerNorm(d_encoder)
        if position_type == 'cond' or position_type == 'conv_static':
            self.position = nn.Conv1d(d_encoder, d_encoder, kernel_size=5, padding='same')
            self.a = nn.Parameter(torch.tensor(1.))
        elif position_type == 'relative':
            self.position = PositionalEmbedding(self.max_len, d_encoder)
        else:
            self.position = PositionalEmbedding(self.max_len, d_encoder, grad=False)

        self.TRMs = nn.ModuleList([
            TransformerBlock(d_encoder, attn_heads, 4 * d_encoder, enable_res_parameter, tr, data_shape[0]) for i in
            range(layers)
        ])

    def forward(self, x):
        x = x.float()
        if len(x.shape) == 4:
            x = x.squeeze(1)
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_norm(x)
        if self.position_location == 'top':
            if self.position_type == 'cond' or self.position_type == 'conv_static':
                x = x.transpose(2, 1)
                if self.position_type == 'cond':
                    x = x + self.position(x)
                else:
                    with torch.no_grad():
                        x = x + self.position(x)
                x = x.transpose(2, 1)
            elif self.position_type != 'none':
                x += self.position(x)
        for index, TRM in enumerate(self.TRMs):
            x = TRM(x, mask=None)
            if index == 1 and self.position_location == 'middle':
                if self.position_type == 'cond':
                    x = x.transpose(2, 1)
                    x = x + self.position(x)
                    x = x.transpose(2, 1)
                elif self.position_type != 'none':
                    x += self.position(x)
        return x


class FormerTime(nn.Module):
    """
    FormerTime model
    """

    def __init__(self, args):
        super(FormerTime, self).__init__()
        attn_heads = args.attn_heads
        layers = args.stages
        enable_res_parameter = args.enable_res_parameter
        num_class = args.num_class

        self.device = args.device
        self.position = args.position_location
        self.pooling_type = args.pooling_type
        self.data_shape = args.data_shape
        self.d_encoder = args.hidden_size_per_stage
        self.slice_sizes = [(i, j) for i, j in zip(args.slice_per_stage, [self.data_shape[1]] + self.d_encoder)]
        self.tr = args.tr
        self.stride = args.stride_per_stage
        self.layer_per_stage = args.layer_per_stage

        self._form_data_shape()
        self.encs = nn.ModuleList([
            Encoder(slice_size=self.slice_sizes[i], data_shape=self.data_shapes[i], d_encoder=self.d_encoder[i],
                       attn_heads=attn_heads, device=self.device, enable_res_parameter=enable_res_parameter,
                       stride=self.stride[i], tr=self.tr[i], layers=self.layer_per_stage[i],
                       position_location=self.position, position_type=args.position_type)
            for i in range(layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(self.data_shapes[-1][0] * self.d_encoder[-1], num_class),
            # nn.Sigmoid()
        ) if self.pooling_type == 'cat' else nn.Sequential(
            nn.Linear(self.d_encoder[-1], num_class),
            # nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def _form_data_shape(self):
        self.data_shapes = []
        for i in range(len(self.tr)):
            if not i:
                data_shape_pre = self.data_shape
            else:
                data_shape_pre = self.data_shapes[-1]
            len_raw = (data_shape_pre[0] - self.slice_sizes[i][0]) // self.stride[i] + 1
            self.data_shapes.append(
                (len_raw, self.d_encoder[i]))
        print(self.data_shapes)

    def forward(self, x):
        for Encs in self.encs:
            x = Encs(x)
        if self.pooling_type == 'last_token':
            return self.output(x[:, -1, :])
        elif self.pooling_type == 'mean':
            return self.output(torch.mean(x, dim=1))
        elif self.pooling_type == 'cat':
            return self.output(x.view(x.shape[0], -1))
        else:
            return self.output(torch.max(x, dim=1)[0])

    def encode(self, x):
        for Encs in self.encs:
            x = Encs(x)
        if self.pooling_type == 'last_token':
            return x[:, -1, :]
        elif self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pooling_type == 'cat':
            return x.view(x.shape[0], -1)
        else:
            return torch.max(x, dim=1)[0]
