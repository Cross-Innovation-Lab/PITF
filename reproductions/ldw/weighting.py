import torch
@torch.no_grad()
def label_weights(unimodal,multimodal):
    return (-(unimodal-multimodal.reshape(-1,1)).abs()).softmax(-1)
