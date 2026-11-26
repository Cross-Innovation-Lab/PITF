import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from reproductions.confede_ldw.gate import ModalityGate

ROOT = Path(__file__).resolve().parents[2]

def masked_mean(x, valid):
    return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

def position(length, width, device, dtype):
    t = torch.arange(length, device=device, dtype=dtype)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000.) / width))
    result = torch.zeros(length, width, device=device, dtype=dtype)
    result[:, 0::2] = (t * f).sin()
    result[:, 1::2] = (t * f).cos()
    return result

class TransformerBody(nn.Module):
    def __init__(self, dims, width=128, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.input_width = self.output_width = width
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, width), nn.LayerNorm(width)) for d in dims])
        self.modality = nn.Parameter(torch.randn(3, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, 4*width, dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        for block in self.encoder.layers:
            for p in block.parameters():
                if p.ndim > 1: nn.init.xavier_uniform_(p)
        self.head = nn.Sequential(nn.Linear(3*width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def representations(self, xs, masks):
        projected = [p(x).masked_fill(~m.unsqueeze(-1), 0) for p, x, m in zip(self.projections, xs, masks)]
        mono = torch.stack([masked_mean(x, m) for x, m in zip(projected, masks)], 1)
        tokens = [x + position(x.shape[1], x.shape[2], x.device, x.dtype) + self.modality[i]
                  for i, x in enumerate(projected)]
        encoded = self.encoder(torch.cat(tokens, 1), src_key_padding_mask=~torch.cat(masks, 1))
        chunks = encoded.split([x.shape[1] for x in xs], 1)
        return mono, torch.stack([masked_mean(x, m) for x, m in zip(chunks, masks)], 1)

    def predict(self, readouts):
        return self.head(readouts.flatten(1)).reshape(-1)

def load_mult_class():
    source = ROOT / 'third_party' / 'Multimodal-Transformer'
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('pi_ldw_official_mult', source / 'src/models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MULTModel

def compatible_position_forward(self, inputs):
    from modules.position_embedding import make_positions
    batch, length = inputs.size()
    maximum = self.padding_idx + 1 + length
    device = inputs.get_device()
    if device not in self.weights or maximum > self.weights[device].size(0):
        self.weights[device] = self.get_embedding(maximum, self.embedding_dim, self.padding_idx)
    self.weights[device] = self.weights[device].type_as(self._float_tensor)
    positions = make_positions(inputs, self.padding_idx, self.left_pad)
    return self.weights[device].index_select(0, positions.reshape(-1)).reshape(batch, length, -1).detach()

class PaddingAttention(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        for name, parameter in upstream._parameters.items(): self.register_parameter(name, parameter)
        for name, module in upstream._modules.items(): self.add_module(name, module)
        for name in ('bias_k', 'bias_v'):
            if not hasattr(self, name): setattr(self, name, getattr(upstream, name))
        for name in ('embed_dim', 'num_heads', 'head_dim', 'scaling', 'attn_dropout', 'add_zero_attn'):
            setattr(self, name, getattr(upstream, name))
        object.__setattr__(self, '_original_forward', type(upstream).forward)
        for name in ('in_proj_qkv', 'in_proj_q', 'in_proj_kv', 'in_proj_k', 'in_proj_v', '_in_proj'):
            if hasattr(type(upstream), name):
                object.__setattr__(self, name, getattr(type(upstream), name).__get__(self, type(self)))
        self.padding_mask = None

    def forward(self, query, key, value, attn_mask=None):
        if self.padding_mask is None or not bool(self.padding_mask.any()):
            return self._original_forward(self, query, key, value, attn_mask=attn_mask)
        return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
            self.attn_dropout, self.out_proj.weight, self.out_proj.bias, training=self.training,
            key_padding_mask=self.padding_mask, need_weights=True, attn_mask=attn_mask)

class MulTBody(nn.Module):
    def __init__(self, dims, layers=2, dropout=.1):
        super().__init__()
        options = SimpleNamespace(orig_d_l=dims[0], orig_d_a=dims[1], orig_d_v=dims[2],
            lonly=True, aonly=True, vonly=True, num_heads=5, layers=layers,
            attn_dropout=dropout, attn_dropout_a=dropout, attn_dropout_v=dropout,
            relu_dropout=dropout, res_dropout=dropout, out_dropout=dropout,
            embed_dropout=dropout, attn_mask=False, output_dim=1)
        self.native = load_mult_class()(options)
        self.input_width, self.output_width = 30, 60
        for name, buffer in self.native.named_buffers():
            if name.endswith('embed_positions._float_tensor'): buffer.zero_()
        for name, module in self.native.named_modules():
            if name.endswith('embed_positions'):
                module.forward = MethodType(compatible_position_forward, module)
        for name, module in list(self.native.named_modules()):
            if name.endswith('self_attn'):
                parent_name, child = name.rsplit('.', 1)
                self.native.get_submodule(parent_name)._modules[child] = PaddingAttention(module)

    @staticmethod
    def attend(network, query, source, source_mask):
        for layer in network.layers: layer.self_attn.padding_mask = ~source_mask
        try:
            return network(query) if source is None else network(query, source, source)
        finally:
            for layer in network.layers: layer.self_attn.padding_mask = None

    def representations(self, xs, masks):
        n = self.native
        xs = [F.dropout(xs[0], p=n.embed_dropout, training=self.training), xs[1], xs[2]]
        ps = [proj(x.transpose(1, 2)).transpose(1, 2).masked_fill(~m.unsqueeze(-1), 0)
              for proj, x, m in zip((n.proj_l, n.proj_a, n.proj_v), xs, masks)]
        mono = torch.stack([masked_mean(x, m) for x, m in zip(ps, masks)], 1)
        seq = [x.transpose(0, 1) for x in ps]
        names = ('l', 'a', 'v')
        outputs = []
        for i, target in enumerate(names):
            sources = [j for j in range(3) if j != i]
            streams = [self.attend(getattr(n, f'trans_{target}_with_{names[j]}'), seq[i], seq[j], masks[j]) for j in sources]
            joined = torch.cat(streams, -1)
            memory = self.attend(getattr(n, f'trans_{target}_mem'), joined, None, masks[i]).transpose(0, 1)
            last = masks[i].sum(1).long() - 1
            outputs.append(memory[torch.arange(len(last), device=last.device), last])
        return mono, torch.stack(outputs, 1)

    def predict(self, readouts):
        n = self.native
        x = readouts.flatten(1)
        residual = n.proj2(F.dropout(F.relu(n.proj1(x)), p=n.out_dropout, training=self.training))
        return n.out_layer(x + residual).reshape(-1)

class PerceiverIOBody(nn.Module):
    def __init__(self, dims, width=128, dropout=.1):
        super().__init__()
        from transformers import PerceiverConfig, PerceiverModel
        from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder
        self.input_width = self.output_width = width
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, width), nn.LayerNorm(width)) for d in dims])
        self.modality = nn.Parameter(torch.randn(3, width) * .02)
        config = PerceiverConfig(d_model=width, d_latents=width, num_latents=64,
            num_blocks=1, num_self_attends_per_block=4, num_self_attention_heads=4,
            num_cross_attention_heads=4, qk_channels=width, v_channels=width,
            self_attention_widening_factor=4, cross_attention_widening_factor=4,
            attention_probs_dropout_prob=dropout, use_query_residual=True)
        self.encoder = PerceiverModel(config)
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=width,
            position_encoding_type='none', num_channels=width, qk_channels=width,
            v_channels=width, num_heads=4, widening_factor=4, use_query_residual=True)
        self.head = nn.Sequential(nn.Linear(3*width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def representations(self, xs, masks):
        ps = [p(x).masked_fill(~m.unsqueeze(-1), 0) for p, x, m in zip(self.projections, xs, masks)]
        mono = torch.stack([masked_mean(x, m) for x, m in zip(ps, masks)], 1)
        tokens = [x + position(x.shape[1], x.shape[2], x.device, x.dtype) + self.modality[i] for i, x in enumerate(ps)]
        latent = self.encoder(inputs=torch.cat(tokens, 1), attention_mask=torch.cat(masks, 1).long()).last_hidden_state
        readouts = self.decoder(mono + self.modality.unsqueeze(0), latent).logits
        return mono, readouts

    def predict(self, readouts):
        return self.head(readouts.flatten(1)).reshape(-1)

class TransferModel(nn.Module):
    def __init__(self, backbone, dims, arm):
        super().__init__()
        if arm not in ('A', 'C'): raise ValueError(arm)
        self.body = {'transformer': TransformerBody, 'mult': MulTBody, 'perceiver_io': PerceiverIOBody}[backbone](dims)
        self.arm = arm
        if arm == 'C': self.gate = ModalityGate(self.body.input_width, hidden=64)

    def forward(self, inputs, force_weights=None):
        xs = [inputs[m] for m in 'TAV']
        masks = [inputs[m + '_valid'] for m in 'TAV']
        mono, readouts = self.body.representations(xs, masks)
        gate = self.gate(mono) if self.arm == 'C' else dict(weights=mono.new_full((len(mono), 3), 1/3))
        weights = gate['weights'] if force_weights is None else force_weights
        prediction = self.body.predict(readouts * (3 * weights).unsqueeze(-1))
        return dict(prediction=prediction, weights=weights, log_weights=gate.get('log_weights'))
