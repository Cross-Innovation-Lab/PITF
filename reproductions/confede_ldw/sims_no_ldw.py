import torch
from torch import nn
from reproductions.confede.modules import encode
from .intensity_contrast import IntensityContrastJoint


class SIMSNoLDWFusion(IntensityContrastJoint):
    def __init__(self, original_cl_off):
        nn.Module.__init__(self)
        if original_cl_off.ntxent_loss.mode != 'off':
            raise ValueError('Expected the historical CL-off backbone')
        self.backbone = original_cl_off.backbone
        self.enabled = False
        self.coefficient = 0.
        self.gate = None

    def set_train(self, train_module=None):
        self.backbone.set_train(train_module)

    def features(self, sample, device):
        t, v, a = encode(self.backbone, sample, device)
        blocks = torch.stack([
            self.T_simi_proj(t), self.V_simi_proj(v), self.A_simi_proj(a),
            self.T_dissimi_proj(t), self.V_dissimi_proj(v), self.A_dissimi_proj(a)
        ], 1)
        weights = t.new_full((len(t), 3), 1 / 3)
        return ((t, v, a), blocks,
                dict(weights=weights, log_weights=weights.log()), blocks.flatten(1))
