import torch
import math
from parlai.agents.transformer.modules import MultiHeadAttention, \
    TransformerEncoderLayer, _normalize, LAYER_NORM_EPS, \
    create_position_codes, gelu
from torch.nn import LayerNorm


class MultiHeadAttentionPast(MultiHeadAttention):
    def forward(self, hidden_states, attention_mask=None, layer_past=None):
        batch_size, query_len, dim = hidden_states.size()
        assert (dim == self.dim), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
            return tensor

        q = self.q_lin(hidden_states)
        k = self.k_lin(hidden_states)
        v = self.v_lin(hidden_states)

        if layer_past is not None:
            past_k, past_v = layer_past[0], layer_past[1]
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = torch.stack([k, v])
        q = prepare_head(q)
        k = prepare_head(k)
        v = prepare_head(v)

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        if layer_past is None and attention_mask is not None:
            dot_prod += attention_mask

        attn_weights = torch.softmax(dot_prod, dim=-1, dtype=torch.float).type_as(hidden_states)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(hidden_states)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, query_len, dim)
        )
        out = self.out_lin(attentioned)
        return out, present


class TransformerEncoderLayerPast(TransformerEncoderLayer):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation='relu',
            variant=None,
    ):
        super().__init__(n_heads, embedding_size, ffn_size, attention_dropout,
                         relu_dropout, dropout, activation, variant)
        self.attention = MultiHeadAttentionPast(n_heads, embedding_size, dropout=attention_dropout)

    def forward(self, tensor, attention_mask=None, layer_past=None):
        """
        Forward pass.
        """

        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm1)
        attended_tensor, layer_past = self.attention(tensor, attention_mask, layer_past)
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm1)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm2)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm2)
        # tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor, layer_past


class TransformerAREncoder(torch.nn.Module):
    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            embedding=None,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            padding_idx=0,
            learn_positional_embeddings=False,
            embeddings_scale=False,
            n_positions=1024,
            activation='relu',
            variant='aiayn',
            n_segments=0,
            output_scaling=1.0,
    ):
        super(TransformerAREncoder, self).__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = dropout
        self.dropout = torch.nn.Dropout(p=self.dropout_frac)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
                embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                    embedding_size is None or embedding_size ==
                    embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )

        # create the positional embeddings
        self.position_embeddings = torch.nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(n_positions, embedding_size, out=self.position_embeddings.weight)
        else:
            torch.nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm' or self.variant == 'prelayernorm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = torch.nn.Embedding(self.n_segments, self.dim)

        # build the model
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayerPast(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                )
            )
        self.output_scaling = output_scaling

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                segments=None, past=None, inputs_embeds=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.layers)
        else:
            past_length = past[0][0].size(-2)

        if input_ids is None:
            assert inputs_embeds is not None
            input_shape = inputs_embeds.size()[:2]
            device = inputs_embeds.device
        else:
            input_shape = input_ids.size()
            device = input_ids.device
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length + past_length), device=device)  # (bs, seq_length)
        seq_ids = torch.arange(past_length + seq_length, device=device)
        # add a upper triangle mask for auto-regressive language model
        causal_mask = seq_ids[None, None, :].repeat(1, past_length + seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, :, :] * attention_mask[:, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        tensor = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if self.embeddings_scale:
            tensor = tensor * math.sqrt(self.dim)

        assert position_ids.max().item() <= self.n_positions
        position_embs = self.position_embeddings(position_ids)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(position_ids)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)
        presents = ()
        for i in range(self.n_layers):
            layer_past = past[i]
            tensor, layer_present = self.layers[i](tensor, extended_attention_mask, layer_past)
            presents = presents + (layer_present,)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)
        tensor *= self.output_scaling
        return tensor, presents


class LMPredictionHead(torch.nn.Module):
    def __init__(self, opt, vocab_size):
        super().__init__()
        hidden_size = opt['embedding_size']
        activation = opt['activation']
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        if activation == 'relu':
            self.nonlinear = torch.relu
        elif activation == 'gelu':
            self.nonlinear = gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.LayerNorm = LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.nonlinear(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
