import torch
from torch import nn
from torch.nn import functional as F

from reproductions.confede.modules import encode
from .component_ablation import component_values
from .difference_geometry import DifferenceGeometryJoint, difference_values


GEOMETRIES = ('absolute', 'polarity_intensity', 'polarity_only', 'intensity_only')


def teacher_distances(unimodal, overall, geometry):
    if geometry in ('polarity_only', 'intensity_only'):
        return component_values(unimodal, overall, geometry)
    if geometry in ('absolute', 'polarity_intensity'):
        return difference_values(unimodal, overall, geometry)
    raise ValueError(f'Unknown trust geometry: {geometry}')


class TrustGuidedAdaptiveFusion(nn.Module):
    def __init__(self, width, hidden=64):
        super().__init__()
        pair_width = 2 * width
        self.feature_gates = nn.ModuleList(
            nn.Sequential(nn.Linear(pair_width, hidden), nn.ReLU(), nn.Linear(hidden, pair_width))
            for _ in 'TAV'
        )
        self.alpha_predictor = nn.Linear(pair_width, 1)
        for layer in [*(gate[-1] for gate in self.feature_gates), self.alpha_predictor]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pairs, trust):
        gated = torch.stack([
            gate(pair).sigmoid() * pair
            for gate, pair in zip(self.feature_gates, pairs.unbind(dim=1))
        ], dim=1)
        pooled = (trust.unsqueeze(-1) * gated).sum(dim=1)
        shared, specific = pooled.chunk(2, dim=-1)
        alpha = self.alpha_predictor(pooled).sigmoid()
        return alpha * shared + (1 - alpha) * specific, alpha


class PITFFusion(DifferenceGeometryJoint):
    def __init__(self, original, temperature=1.0, geometry='polarity_intensity'):
        if geometry not in GEOMETRIES:
            raise ValueError(f'Unknown trust geometry: {geometry}')
        if temperature <= 0:
            raise ValueError('Trust temperature must be positive')
        super().__init__(original, 'polarity_intensity', temperature=temperature)
        self.difference_mode = geometry
        decoder = self.backbone.TVA_decoder.MLP
        first = decoder[0]
        width = first.in_features // 6
        with torch.random.fork_rng(devices=[]):
            self.adaptive_fusion = TrustGuidedAdaptiveFusion(width).to(first.weight)
            decoder[0] = nn.Linear(width, first.out_features).to(first.weight)

    def features(self, sample, device):
        t, v, a = encode(self.backbone, sample, device)
        shared = [self.T_simi_proj(t), self.V_simi_proj(v), self.A_simi_proj(a)]
        specific = [self.T_dissimi_proj(t), self.V_dissimi_proj(v), self.A_dissimi_proj(a)]
        blocks = torch.stack(shared + specific, dim=1)
        pairs = torch.stack([
            torch.cat((shared[index], specific[index]), dim=-1) for index in (0, 2, 1)
        ], dim=1)
        weights = self.gate(pairs)
        fused, weights['fusion_alpha'] = self.adaptive_fusion(pairs, weights['weights'])
        return (t, v, a), blocks, weights, fused

    def forward(self, sample1, sample2=None, return_loss=True, device=None):
        device = next(self.parameters()).device if device is None else device
        if not return_loss:
            return self.infer(sample1, device)

        samples = (sample1,) if sample2 is None else (sample1, sample2)
        features = [self.features(sample, device) for sample in samples]
        truth = [sample['regression_labels'].to(device).float().reshape(-1) for sample in samples]
        unimodal = [
            torch.stack([sample[f'regression_labels_{m}'].to(device).float().reshape(-1) for m in 'TAV'], dim=1)
            for sample in samples
        ]
        auxiliary_truth = torch.cat([
            torch.cat([overall, overall, overall, mono[:, 0], mono[:, 2], mono[:, 1]])
            for overall, mono in zip(truth, unimodal)
        ])
        auxiliary_features = torch.cat([torch.cat(blocks.unbind(dim=1), dim=0) for _, blocks, _, _ in features])
        prediction = self.TVA_decoder(torch.cat([fused for _, _, _, fused in features])).reshape(-1)
        overall = torch.cat(truth)
        main_loss = F.mse_loss(prediction, overall)
        auxiliary_loss = F.mse_loss(self.mono_decoder(auxiliary_features).reshape(-1), auxiliary_truth)
        trust_loss = main_loss.new_zeros(())
        if self.coefficient:
            distances = teacher_distances(torch.cat(unimodal), overall, self.difference_mode)
            target = (-distances / self.temperature).softmax(-1)
            log_weights = torch.cat([weights['log_weights'] for _, _, weights, _ in features])
            trust_loss = F.kl_div(log_weights, target, reduction='batchmean')
        loss = main_loss + .01 * auxiliary_loss + self.coefficient * trust_loss
        return prediction, features[0][0], loss, main_loss, main_loss.new_zeros(())
