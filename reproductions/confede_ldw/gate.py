import torch
from torch import nn


class ModalityGate(nn.Module):
    def __init__(self, input_dim=768, hidden=64, kind='dynamic', normalize_inputs=False, learn_strength=False):
        super().__init__()
        if kind not in ('uniform', 'static', 'dynamic'):
            raise ValueError(kind)
        self.input_dim, self.hidden, self.kind = input_dim, hidden, kind
        self.options = {'normalize_inputs':normalize_inputs, 'learn_strength':learn_strength}
        self.input_normalizer = nn.LayerNorm(input_dim, elementwise_affine=False) if normalize_inputs else nn.Identity()
        if learn_strength:
            self.strength_logit = nn.Parameter(torch.zeros(()))
        if kind == 'dynamic':
            self.network = nn.Sequential(nn.Linear(3*input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        elif kind == 'static':
            self.logits = nn.Parameter(torch.zeros(3))

    def forward(self, encoded):
        if encoded.ndim != 3 or encoded.shape[1:] != (3, self.input_dim):
            raise ValueError(f'Expected [batch,3,{self.input_dim}] T/A/V features')
        if self.kind == 'dynamic':
            logits = self.network(self.input_normalizer(encoded).flatten(1))
        elif self.kind == 'static':
            logits = self.logits.unsqueeze(0).expand(len(encoded), -1)
        else:
            logits = encoded.new_zeros((len(encoded), 3))
        output = {'weights': logits.softmax(-1), 'log_weights': logits.log_softmax(-1)}
        if self.options['learn_strength']:
            output['strength'] = self.strength_logit.sigmoid()
        return output


def apply_modality_gains(projected, weights, branch='all', strength=None):
    if branch not in ('all', 'specific'):
        raise ValueError(branch)
    if projected.ndim != 3 or projected.shape[1] != 6:
        raise ValueError('Expected [batch,6,width]: sT,sV,sA,dT,dV,dA')
    if weights.shape != (len(projected), 3):
        raise ValueError('Expected T/A/V weights [batch,3]')
    gains = (3*weights)[:, [0, 2, 1, 0, 2, 1]]
    if strength is not None:
        gains = 1+strength*(gains-1)
    if branch == 'specific':
        gains = torch.cat([torch.ones_like(gains[:, :3]), gains[:, 3:]], dim=1)
    return (projected*gains.unsqueeze(-1)).flatten(1)
