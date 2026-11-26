from __future__ import annotations

import copy
import torch
from torch import nn


def encode(model, batch, device):
    return (model.text_encoder(batch['raw_text'], device=device).flatten(1),
            model.vision_encoder(batch['vision'].to(device), batch['vision_padding_mask'].to(device), device=device).flatten(1),
            model.audio_encoder(batch['audio'].to(device), batch['audio_padding_mask'].to(device), device=device).flatten(1))


def predict_full(model, batch, device):
    t,v,a = encode(model, batch, device)
    similar = [model.T_simi_proj(t), model.V_simi_proj(v), model.A_simi_proj(a)]
    specific = [model.T_dissimi_proj(t), model.V_dissimi_proj(v), model.A_dissimi_proj(a)]
    fused = model.TVA_decoder(torch.cat(similar + specific, dim=-1)).reshape(-1)
    mono = model.mono_decoder(torch.cat(similar + specific, dim=0)).reshape(6, len(t))
    return {'M': fused, 'T': mono[3], 'V': mono[4], 'A': mono[5]}


class UnimodalPredictor(nn.Module):

    def __init__(self, encoder, projector, head, modality):
        super().__init__()
        self.encoder, self.projector, self.head = encoder, projector, head
        self.modality = modality

    def forward(self, value, padding_mask=None):
        device = next(self.parameters()).device
        if self.modality == 'T':
            features = self.encoder(value, device=device).flatten(1)
        else:
            value = torch.as_tensor(value,device=device,dtype=torch.float32)
            value = torch.nan_to_num(value,nan=0.0,posinf=0.0,neginf=0.0)
            if value.ndim != 3:
                raise ValueError('Audio/vision features must have shape [batch,time,feature]')
            if padding_mask is None:
                padding_mask = value.sum(dim=-1) == 0
                padding_mask[:,0] = False
                padding_mask = torch.cat([torch.zeros_like(padding_mask[:,:1]),padding_mask],dim=1)
            padding_mask = torch.as_tensor(padding_mask,device=device,dtype=torch.bool)
            features = self.encoder(value, padding_mask, device=device).flatten(1)
        return self.head(self.projector(features)).reshape(-1)


def extract_modality(model, modality, clone=True):
    names = {'T':'text', 'A':'audio', 'V':'vision'}
    pieces = [getattr(model,names[modality]+'_encoder'), getattr(model,modality+'_dissimi_proj'), model.mono_decoder]
    if clone:
        pieces = [copy.deepcopy(piece) for piece in pieces]
    return UnimodalPredictor(*pieces, modality).eval()


def conflict_from_scores(text, audio, vision, neutral_margin=.1):
    values = [torch.where(x.abs() <= neutral_margin, 0, x) for x in (text,audio,vision)]
    t,a,v = values
    return ((t*a)<0) | ((t*v)<0)


def load_predictor(checkpoint, device='cuda:0'):
    from pathlib import Path
    from reproductions.confede.runtime import configure
    payload = torch.load(checkpoint, map_location='cpu')
    cfg = configure(Path(checkpoint).resolve().parent, payload['dataset'])
    cfg.DEVICE = torch.device(device)
    from model.net.constrastive.text_encoder_finetune import TextEncoder
    from model.net.constrastive.audio_encoder_fintune import AudioEncoder
    from model.net.constrastive.vision_encoder_finetune import VisionEncoder
    from model.net.constrastive.TVA_fusion import projector
    from model.decoder.classifier import BaseClassifier
    modality = payload['modality']
    if modality == 'T':
        encoder = TextEncoder(with_projector=False, config=cfg)
    else:
        encoder = {'A':AudioEncoder,'V':VisionEncoder}[modality](config=cfg)
    width = cfg.SIMS.downStream.encoder_fea_dim
    model = UnimodalPredictor(encoder, projector(width,width//2),
                             BaseClassifier(width//2,[width//4,width//8],1),modality)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval()
