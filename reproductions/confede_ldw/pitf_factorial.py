import torch
from torch import nn

from .gate import ModalityGate
from .pitf import PITFFusion


ARMS = tuple(f'{code:03b}' for code in range(8))


class FixedHalfGate(nn.Module):

    def __init__(self, scalar=False):
        super().__init__()
        self.scalar = scalar

    def forward(self, features):
        return torch.zeros_like(features[..., :1] if self.scalar else features)


def build_factorial_pitf(original, temperature, arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown PITF ablation variant: {arm}')
    model = PITFFusion(original, temperature)
    trust, feature, mixing = (flag == '1' for flag in arm)
    if not trust:
        model.gate = ModalityGate(model.gate.input_dim, kind='uniform')
        model.enabled = False
        model.coefficient = 0.0
    if not feature:
        model.adaptive_fusion.feature_gates = nn.ModuleList(FixedHalfGate() for _ in 'TAV')
    if not mixing:
        model.adaptive_fusion.alpha_predictor = FixedHalfGate(scalar=True)
    return model
