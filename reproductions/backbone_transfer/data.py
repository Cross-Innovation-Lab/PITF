import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

from reproductions.confede.data import SIMSData
from reproductions.confede.runtime import save_json, sha256
from reproductions.confede_ldw.difference_geometry import (
    difference_values,
    fit_entropy_temperature,
    mean_entropy,
)
from reproductions.confede_ldw.unic_data import UniCData


CAP = 64
SPLITS = ('train', 'valid', 'test')


def raw_dataset(name, features, annotations, split):
    if name == 'unic':
        return UniCData(features, split, annotations=annotations)
    return SIMSData(features, split, dataset=name)


def pool_sequence(array, cap=CAP):
    array = np.asarray(array, dtype=np.float32)
    if not len(array):
        array = np.zeros((1, array.shape[-1]), dtype=np.float32)
    if len(array) > cap:
        edges = np.linspace(0, len(array), cap + 1).astype(int)
        array = np.stack([
            array[left:right].mean(0, dtype=np.float64)
            for left, right in zip(edges[:-1], edges[1:])
        ]).astype(np.float32)
    valid = np.arange(cap) < len(array)
    output = np.zeros((cap, array.shape[-1]), dtype=np.float32)
    output[:len(array)] = array
    return output, valid


def av_sequence(data, mode, index):
    key = {'A': 'audio', 'V': 'vision'}[mode]
    frames = np.array(data[key][index], dtype=np.float32, copy=True)
    np.nan_to_num(frames, copy=False, nan=0, posinf=0, neginf=0)
    if key + '_lengths' in data:
        length = int(data[key + '_lengths'][index])
        if not 0 <= length <= len(frames):
            raise ValueError(f'Invalid {key} length for row {index}: {length}')
        frames = frames[:length]
    else:
        frames = frames[np.any(frames != 0, axis=-1)]
    return pool_sequence(frames)


def load_cache(directory):
    metadata = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
    for relative, digest in metadata['files_sha256'].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f'Cached sequence file changed: {directory / relative}')
    return metadata


def write_split(folder, dataset, tokenizer, encoder, device):
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(dataset.data['id']).astype(str)
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate sample IDs in {dataset.split}')
    dims = [encoder.config.hidden_size, dataset.data['audio'].shape[-1],
            dataset.data['vision'].shape[-1]]
    arrays = {
        mode: np.lib.format.open_memmap(
            folder / (mode + '.npy'), mode='w+', dtype='float32',
            shape=(len(dataset), CAP, width),
        )
        for mode, width in zip('TAV', dims)
    }
    masks = {
        mode: np.lib.format.open_memmap(
            folder / (mode + '_valid.npy'), mode='w+', dtype='bool',
            shape=(len(dataset), CAP),
        )
        for mode in 'TAV'
    }
    for start in range(0, len(dataset), 16):
        stop = min(start + 16, len(dataset))
        text = [str(value) for value in dataset.data['raw_text'][start:stop]]
        batch = tokenizer(text, padding=True, truncation=True, max_length=256,
                          return_tensors='pt').to(device)
        with torch.inference_mode():
            hidden = encoder(**batch).last_hidden_state.cpu().numpy()
        valid = batch['attention_mask'].cpu().numpy().astype(bool)
        for row, index in enumerate(range(start, stop)):
            arrays['T'][index], masks['T'][index] = pool_sequence(hidden[row, valid[row]])
            for mode in 'AV':
                arrays[mode][index], masks[mode][index] = av_sequence(dataset.data, mode, index)
    for value in (*arrays.values(), *masks.values()):
        value.flush()
    labels = np.stack([dataset.labels[mode] for mode in 'MTAV'], 1).astype(np.float32)
    if not np.isfinite(labels).all() or np.max(np.abs(labels)) > 1 + 1e-6:
        raise ValueError(f'Expected finite MTAV labels between -1 and 1 in {dataset.split}')
    np.save(folder / 'ids.npy', ids)
    np.save(folder / 'labels_MTAV.npy', labels)
    return dims, set(ids)


def normalize_audio_vision(directory):
    for mode in 'AV':
        array_name = mode + '.npy'
        mask_name = mode + '_valid.npy'
        train = np.load(directory / 'train' / array_name, mmap_mode='r')
        mask = np.load(directory / 'train' / mask_name, mmap_mode='r')
        count = 0
        sums = np.zeros(train.shape[-1])
        squares = np.zeros(train.shape[-1])
        for start in range(0, len(train), 32):
            frames = train[start:start + 32][mask[start:start + 32]].astype(np.float64)
            count += len(frames)
            sums += frames.sum(0)
            squares += (frames * frames).sum(0)
        mean = sums / count
        std = np.sqrt(np.maximum(0, squares / count - mean * mean)).clip(1e-4)
        np.savez(directory / (mode + '_train_statistics.npz'), mean=mean, std=std, count=count)
        del train, mask
        for split in SPLITS:
            array = np.load(directory / split / array_name, mmap_mode='r+')
            valid = np.load(directory / split / mask_name, mmap_mode='r')
            for start in range(0, len(array), 32):
                rows = slice(start, start + 32)
                frames = np.clip((array[rows].astype(np.float64) - mean) / std, -5, 5)
                frames = frames.astype(np.float32)
                frames[~valid[rows]] = 0
                array[rows] = frames
            array.flush()


def prepare_cache(directory, name, features, annotations, bert, checkpoint_path, device):
    directory = Path(directory)
    if (directory / 'complete.json').is_file():
        return load_cache(directory)
    tokenizer = BertTokenizer.from_pretrained(bert, local_files_only=True)
    encoder = BertModel.from_pretrained(bert, local_files_only=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    prefix = 'encoder.extractor.'
    state = {
        key.removeprefix(prefix): value for key, value in checkpoint['model'].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f'Expected a ConFEDE text encoder checkpoint at {checkpoint_path}')
    encoder.load_state_dict(state, strict=True)
    del checkpoint, state
    encoder.to(device).eval().requires_grad_(False)

    width = None
    ids = set()
    counts = {}
    for split in SPLITS:
        dataset = raw_dataset(name, features, annotations, split)
        dims, split_ids = write_split(directory / split, dataset, tokenizer, encoder, device)
        if width is not None and width != dims:
            raise ValueError(f'Modality dimensions differ in {split}')
        if ids & split_ids:
            raise ValueError(f'Sample IDs overlap with another split in {split}')
        width = dims
        ids.update(split_ids)
        counts[split] = len(dataset)
        print(f'{name} {split}: cached {len(dataset)} samples', flush=True)
        del dataset
    del encoder
    normalize_audio_vision(directory)
    labels = np.load(directory / 'train' / 'labels_MTAV.npy')
    features = torch.as_tensor(labels[:, 1:], dtype=torch.float64)
    target = torch.as_tensor(labels[:, 0], dtype=torch.float64)
    entropy = mean_entropy(difference_values(features, target, 'absolute'), 1.)
    temperature = fit_entropy_temperature(difference_values(features, target, 'polarity_intensity'), entropy)
    files = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted([*directory.rglob('*.npy'), *directory.glob('*.npz')])
    }
    metadata = {
        'dataset': name,
        'dims': width,
        'cap': CAP,
        'split_counts': counts,
        'calibration': {'temperature': temperature, 'target_entropy': entropy, 'split': 'train'},
        'files_sha256': files,
    }
    save_json(directory / 'complete.json', metadata)
    return metadata


class SequenceDataset(Dataset):
    def __init__(self, folder):
        self.features = {mode: np.load(folder / (mode + '.npy'), mmap_mode='r') for mode in 'TAV'}
        self.valid = {
            mode: np.load(folder / (mode + '_valid.npy'), mmap_mode='r') for mode in 'TAV'
        }
        self.labels = np.load(folder / 'labels_MTAV.npy')
        self.ids = np.load(folder / 'ids.npy')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            mode: torch.from_numpy(np.array(values[index], copy=True))
            for mode, values in self.features.items()
        }
        item.update({
            mode + '_valid': torch.from_numpy(np.array(values[index], copy=True))
            for mode, values in self.valid.items()
        })
        item['labels'] = torch.from_numpy(self.labels[index].copy())
        return item
