from transformers import RobertaModel, T5EncoderModel, AutoModel,T5ForConditionalGeneration
import math
from typing import List, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_
import warnings

Tensor = torch.Tensor


# NaiveFourierKANLayer definition (as previously discussed)
class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))
        self.fouriercoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.fouriercoeffs)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        outshape = x.shape[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, gridsize + 1, device=x.device), (1, 1, 1, gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1, :, :, :gridsize], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2, :, :, :gridsize], (-2, -1))
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y


class FSM(nn.Module):
    def __init__(self, num_embeddings, num_channel, embed_dim, filter_sizes, pad_idx=0):
        super(FSM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=num_channel,
                                              kernel_size=(filter_size, embed_dim))
                                    for filter_size in filter_sizes])
        self.re_convs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, len(filter_sizes)),
                                  stride=(1, len(filter_sizes)))

    def forward(self, x):

        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)

        # Apply convolutional layers
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]  # [(batch_size, num_channel, new_seq_len), ...]

        # Max pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [(batch_size, num_channel), ...]

        # Concatenate pooled features
        pooled_features = torch.cat(pooled, dim=1).unsqueeze(0)  # (1, batch_size, num_channel * len(filter_sizes))

        return pooled_features, x.squeeze(1)  # (pooled_features, original_embeddings)

def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # 自注意
            return nn.functional.linear(q, w, b).chunk(3, dim=-1)
        else:
            # seq2seq模型
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (nn.functional.linear(q, w_q, b_q),) + nn.functional.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.functional.linear(q, w_q, b_q), nn.functional.linear(k, w_k, b_k), nn.functional.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)

    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask

    attn = nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = nn.functional.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None,
                 batch_first=False) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        # assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):

    def __init__(self, FSM, encoder_layer, num_layers, filter_sizes, num_channel, output_dim, embed_dim, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.input_dim = len(filter_sizes) * num_channel
        self.fc = nn.Linear(self.input_dim, output_dim)
        self.positional_encoding = FSM(num_embeddings=32100, num_channel=num_channel, embed_dim=embed_dim,
                                       filter_sizes=filter_sizes)
        self.fourierkan1 = NaiveFourierKANLayer(self.input_dim, output_dim, initial_gridsize=2)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = self.positional_encoding(src)[0]

        residual = output

        for _ in range(self.num_layers):
            out = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            out = self.norm(out)

        out = out + residual

        out = self.fc(out.reshape(-1, self.input_dim))
        # out = self.fourierkan1(out.reshape(-1, self.input_dim))
        return out


class SCOPE(nn.Module):
    def __init__(self, d_model, ndead, output_dim, batch_first=True, FSM=FSM):
        super(SCOPE, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=ndead, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(FSM=FSM, encoder_layer=self.encoder_layer, num_layers=6,
                                                      filter_sizes=[2, 3, 5, 7, 11], num_channel=200,
                                                      output_dim=output_dim,
                                                      embed_dim=d_model)

    def forward(self, input_ids, attention_mask=None):
        return self.transformer_encoder(input_ids)


class CodeBert(nn.Module):
    def __init__(self, output_dim):
        super(CodeBert, self).__init__()
        self.encoder = RobertaModel.from_pretrained('./model/CodeBert/')
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        # input_ids = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        input_ids = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.fc(input_ids.mean(dim=1))


class CodeT5(nn.Module):
    def __init__(self, output_dim):
        super(CodeT5, self).__init__()
        self.encoder = AutoModel.from_pretrained('./model/CodeT5/')
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids = input_ids,
                               decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        hidden_states = hidden_states.mean(dim=1)
        return self.fc(hidden_states)


class Unixcoder(nn.Module):
    def __init__(self, output_dim):
        super(Unixcoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('./model/Unixcoder/')
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        input_ids = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        input_ids = input_ids.mean(dim=1)
        return self.fc(input_ids)


class Longformer(nn.Module):
    def __init__(self, output_dim):
        super(Longformer, self).__init__()
        self.encoder = AutoModel.from_pretrained('./model/Longformer/')
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        input_ids = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        input_ids = input_ids.mean(dim=1)
        return self.fc(input_ids)


class Backtracing(nn.Module):
    def __init__(self, num_channel, embed_dim, filter_sizes):
        super(Backtracing, self).__init__()
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=num_channel, out_channels=1, kernel_size=(filter_size, embed_dim))
            for filter_size in filter_sizes
        ])

    def forward(self, pooled_features, original_embeddings):

        batch_size, seq_len, embed_dim = original_embeddings.shape

        pooled_features = pooled_features.view(batch_size, len(self.deconvs), -1,
                                               1)

        backtraced_maps = [F.relu(deconv(pooled_features[:, i, :, :])).squeeze(3) for i, deconv in
                           enumerate(self.deconvs)]

        token_importance = torch.sum(torch.stack(backtraced_maps), dim=0)  # (batch_size, seq_len)

        token_importance = token_importance / (
                    token_importance.sum(dim=1, keepdim=True) + 1e-9)  # Avoid division by zero

        return token_importance


def load_scope_model(scope_model_path,args):

    scope_model = SCOPE(d_model=200, ndead=20, output_dim=args.output_dim)
    scope_model.load_state_dict(torch.load(scope_model_path))
    scope_model.eval()

    fsm = scope_model.transformer_encoder.FSM
    backtracing = Backtracing(num_channel=200, embed_dim=128, filter_sizes=[2, 3, 5, 7, 11])

    return scope_model, fsm, backtracing


def perform_backtracing(scope_model, fsm, backtracing, input_sequence, line_mapping):
    with torch.no_grad():
        _, pooled_features, original_embeddings = fsm(input_sequence)
        token_importance = backtracing(pooled_features, original_embeddings)

        line_aggregator = LineAggregation(line_mapping)
        line_importance = line_aggregator.compute_line_importance(token_importance)

    return line_importance


class LineAggregation:

    def __init__(self, line_mapping):

        self.line_mapping = line_mapping

    def compute_line_importance(self, token_importance):

        batch_size, _ = token_importance.shape
        num_lines = len(self.line_mapping)

        line_importance = torch.zeros(batch_size, num_lines, device=token_importance.device)

        for line_idx, token_indices in enumerate(self.line_mapping):
            line_importance[:, line_idx] = token_importance[:, token_indices].sum(dim=1)

        line_importance = line_importance / (line_importance.sum(dim=1, keepdim=True) + 1e-9)

        return line_importance
