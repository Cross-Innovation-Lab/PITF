from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import Dataset

from triagedmsa.mmap_pickle import load_mmap_pickle


class SIMSData(Dataset):

    def __init__(self, path, split, limit=None, sampler_mode='corrected', dataset='sims'):
        self.split = split
        self.data = load_mmap_pickle(path)[split]
        self.size = len(self.data['regression_labels'])
        if limit is not None:
            self.size = min(self.size, limit)
        self.labels = {m: np.asarray(self.data.get('regression_labels' + ('' if m == 'M' else '_' + m), self.data['regression_labels']))[:self.size].reshape(-1).copy() for m in 'MTAV'}
        self.sampler_mode = sampler_mode
        self.dataset = dataset
        self.pools = None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {'index': index, 'raw_text': str(self.data['raw_text'][index])}
        for key in ['audio', 'vision']:
            value = np.array(self.data[key][index], dtype=np.float32, copy=True)
            np.nan_to_num(value, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            mask = value.sum(axis=-1) == 0
            mask[0] = False
            sample[key] = torch.from_numpy(value)
            sample[key + '_padding_mask'] = torch.from_numpy(np.concatenate(([False], mask)))
        for m in 'MTAV':
            sample['regression_labels' + ('' if m == 'M' else '_' + m)] = torch.tensor(self.labels[m][index], dtype=torch.float32)
        return sample

    def update_neighbors(self, embeddings):
        if self.split != 'train':
            raise ValueError('Only TRAIN may supply contrastive neighbors')
        x = torch.nn.functional.normalize(torch.as_tensor(embeddings).float(), dim=-1, eps=1e-6)
        self.pools = {'ss': [], 'dd': [], 'sd': []}
        dataset = getattr(self, 'dataset', 'sims')
        labels = np.round(self.labels['M']) if dataset in ('mosi','mosei') else self.labels['M']
        cutoff = self.size if dataset=='mosi' else self.size//10 if dataset=='mosei' else self.size//2
        pool_limit = 50 if dataset=='mosei' else 10
        for start in range(0, self.size, 128):
            ranks = torch.argsort(x[start:start+128] @ x.T, dim=1, descending=True).cpu().numpy()
            for offset, row in enumerate(ranks):
                i = start + offset
                near, far = row[:cutoff], row[-1:-cutoff:-1]
                if getattr(self, 'sampler_mode', 'corrected') == 'upstream':
                    near = near[np.arange(cutoff) != i]
                else:
                    near, far = near[near != i], far[far != i]
                choices = {'ss': near[labels[near] == labels[i]],
                           'sd': near[labels[near] != labels[i]],
                           'dd': far[labels[far] != labels[i]]}
                for key, ids in choices.items():
                    if len(ids) < 2:
                        raise ValueError(f'Insufficient {key} neighbors for TRAIN sample {i}')
                    self.pools[key].append(ids[:pool_limit].tolist())

    def sample_neighbors(self, indices):
        from torch.utils.data._utils.collate import default_collate
        if self.pools is None:
            raise RuntimeError('Collect TRAIN embeddings before sampling')
        selected = []
        for i in indices:
            for key in ['ss', 'dd', 'sd']:
                selected.extend(random.sample(self.pools[key][int(i)], 2))
        return default_collate([self[i] for i in selected])


def slice_batch(batch, start, end):
    return {key: value[start:end] for key, value in batch.items()}
