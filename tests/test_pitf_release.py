import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reproductions.confede_ldw import pitf
from reproductions.confede_ldw.intensity_contrast import IntensityContrastJoint
from reproductions.confede_ldw.pitf_factorial import ARMS, build_factorial_pitf


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.TVA_decoder = nn.Sequential(nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 1))
        self.TVA_decoder = TinyClassifier(self.TVA_decoder)
        self.mono_decoder = nn.Linear(4, 1)
        self.ntxent_loss = nn.Identity()
        for modality in 'TAV':
            for branch in ('simi', 'dissimi'):
                setattr(self, f'{modality}_{branch}_proj', nn.Linear(8, 4))


class TinyClassifier(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.MLP = layers

    def forward(self, features):
        return self.MLP(features)


def original_model():
    torch.manual_seed(29)
    return IntensityContrastJoint(TinyBackbone(), contrastive_mode='off', input_dim=8)


@pytest.fixture
def samples(monkeypatch):
    monkeypatch.setattr(pitf, 'encode', lambda _model, sample, _device: sample['embeddings'])
    generator = torch.Generator().manual_seed(12)
    anchor = {
        'embeddings': tuple(torch.randn(3, 8, generator=generator) for _ in 'TVA'),
        'regression_labels': torch.tensor([.5, -.4, 0.]),
        'regression_labels_T': torch.tensor([-.2, -.1, .3]),
        'regression_labels_A': torch.tensor([.3, -.7, -.4]),
        'regression_labels_V': torch.tensor([.1, .3, 0.]),
    }
    partner = {
        key: tuple(-part for part in value) if key == 'embeddings' else -value
        for key, value in anchor.items()
    }
    return anchor, partner


@pytest.mark.parametrize('geometry', pitf.GEOMETRIES)
def test_training_uses_overall_and_unimodal_targets(samples, geometry):
    temperature = 1.0 if geometry == 'absolute' else .7
    model = pitf.PITFFusion(original_model(), temperature, geometry=geometry)
    anchor, partner = samples
    prediction, _, loss, main_loss, contrastive = model(anchor, partner)
    outputs = [model.features(sample, 'cpu') for sample in samples]
    truth = torch.cat([sample['regression_labels'] for sample in samples])
    mono = torch.cat([
        torch.stack([sample[f'regression_labels_{m}'] for m in 'TAV'], dim=1)
        for sample in samples
    ])
    overall = truth[:, None]
    polarity = .5 * ((mono / .1).tanh() - (overall / .1).tanh()).abs()
    intensity = (mono.abs() - overall.abs()).abs()
    distances = {
        'absolute': (mono - overall).abs(),
        'polarity_intensity': polarity + intensity,
        'polarity_only': polarity,
        'intensity_only': intensity,
    }[geometry]
    log_weights = torch.cat([output[2]['log_weights'] for output in outputs])
    expected_kl = F.kl_div(log_weights, (-distances / temperature).softmax(dim=1), reduction='batchmean')
    aux_truth = torch.cat([
        torch.cat([sample['regression_labels']] * 3 + [sample[f'regression_labels_{m}'] for m in 'TVA'])
        for sample in samples
    ])
    aux_features = torch.cat([output[1].transpose(0, 1).flatten(0, 1) for output in outputs])
    expected_aux = F.mse_loss(model.mono_decoder(aux_features).squeeze(-1), aux_truth)
    torch.testing.assert_close(main_loss, F.mse_loss(prediction, truth))
    torch.testing.assert_close(loss, main_loss + .01 * expected_aux + .35 * expected_kl)
    assert contrastive.item() == 0


def test_factorial_initialization_and_gradient_routing(samples):
    anchor, _ = samples
    clean = {'embeddings': anchor['embeddings']}
    reference = build_factorial_pitf(original_model(), .7, '111').infer(clean)['M']
    for arm in ARMS:
        model = build_factorial_pitf(original_model(), .7, arm)
        clean_prediction = model.infer(clean)['M']
        torch.testing.assert_close(clean_prediction, reference, rtol=0, atol=0)
        torch.testing.assert_close(clean_prediction, model.infer(anchor)['M'], rtol=0, atol=0)
        model(anchor)[2].backward()
        parts = (model.gate, model.adaptive_fusion.feature_gates, model.adaptive_fusion.alpha_predictor)
        active = [any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()) for part in parts]
        assert active == [flag == '1' for flag in arm]
