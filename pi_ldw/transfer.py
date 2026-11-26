import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
from statistics import mean, stdev
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from reproductions.backbone_transfer.data import SequenceDataset, load_cache, prepare_cache
from reproductions.backbone_transfer.models import TransferModel
from reproductions.confede.metrics import sentiment_metrics
from reproductions.confede.runtime import replace_atomic, save_json, seed_all, sha256
from reproductions.confede_ldw.difference_geometry import difference_values
from reproductions.confede_ldw.self_stopping import POLICY, recovery_decision


BACKBONES = ('transformer', 'mult', 'perceiver_io')
DATASETS = ('sims', 'simsv2', 'unic')
ARMS = ('A', 'C')
SEEDS = (1, 12, 123, 1234, 1235)
METRICS = ('acc2', 'f1_weighted', 'acc3', 'acc5', 'mae', 'corr')
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
KL_COEFFICIENT = 0.35


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_checkpoint(path, state):
    pending = path.with_suffix('.pending')
    torch.save(state, pending)
    replace_atomic(pending, path)


def save_predictions(path, dataset, prediction, weights):
    pending = path.with_suffix('.pending.npz')
    np.savez_compressed(pending, prediction=prediction, weights=weights,
                        ids=dataset.ids, labels_MTAV=dataset.labels)
    replace_atomic(pending, path)


def body_hash(model):
    digest = hashlib.sha256()
    for key, value in model.body.state_dict().items():
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state(device):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def inputs(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key != 'labels'}


def selection(metrics, epoch):
    acc2, f1 = metrics['acc2'], metrics['f1_weighted']
    return min(acc2, f1), acc2 + f1, -metrics['mae'], -epoch


def training_loss(model, batch, temperature, device):
    result = model(inputs(batch, device))
    labels = batch['labels'].to(device)
    mse = F.mse_loss(result['prediction'], labels[:, 0])
    kl = mse.new_zeros(())
    if model.arm == 'C':
        distances = difference_values(labels[:, 1:], labels[:, 0], 'polarity_intensity')
        teacher = (-distances / temperature).softmax(-1)
        kl = F.kl_div(result['log_weights'], teacher, reduction='batchmean')
    return mse + KL_COEFFICIENT * kl, mse, kl


@torch.no_grad()
def evaluate(model, dataset, name, device):
    model.eval()
    predictions = []
    weights = []
    for batch in DataLoader(dataset, batch_size=BATCH_SIZE):
        result = model(inputs(batch, device))
        predictions.append(result['prediction'].cpu().numpy())
        weights.append(result['weights'].cpu().numpy())
    prediction = np.concatenate(predictions)
    weight = np.concatenate(weights)
    return sentiment_metrics(dataset.labels[:, 0], prediction, name), prediction, weight


def model_folder(output, backbone, arm, seed):
    return output / backbone / arm / f'seed_{seed}'


def train_one(output, cache, name, backbone, arm, seed, metadata, device):
    folder = model_folder(output, backbone, arm, seed)
    folder.mkdir(parents=True, exist_ok=True)
    selected_path = folder / 'best.pt'
    finished_path = folder / 'complete.json'
    resume_path = folder / 'last.pt'
    if finished_path.is_file():
        if sha256(selected_path) != read_json(finished_path)['best_sha256']:
            raise ValueError(f'Checkpoint digest changed: {selected_path}')
        return

    seed_all(seed)
    model = TransferModel(backbone, metadata['dims'], arm).to(device)
    initialization = {'body_init_sha256': body_hash(model)}
    if arm == 'C':
        paired = model_folder(output, backbone, 'A', seed) / 'initialization.json'
        if initialization != read_json(paired):
            raise ValueError(f'A/C initialization differs for {backbone} seed {seed}')
    save_json(folder / 'initialization.json', initialization)

    seed_all(1000 + seed)
    generator = torch.Generator().manual_seed(2000 + seed)
    train = SequenceDataset(cache / 'train')
    valid = SequenceDataset(cache / 'valid')
    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(loader), MAX_EPOCHS * len(loader))
    history, best, start = [], None, 0
    if resume_path.is_file():
        state = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, best, start = state['history'], state['best'], state['epoch']
        generator.set_state(state['loader_rng'])
        restore_rng(state['rng'])
        del state

    temperature = metadata['calibration']['temperature']
    for epoch in range(start + 1, MAX_EPOCHS + 1):
        if recovery_decision(history)['stop']:
            break
        started = time.monotonic()
        model.train()
        sums = np.zeros(3)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total, mse, kl = training_loss(model, batch, temperature, device)
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite loss in {folder}, epoch {epoch}')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            sums += np.array([float(x.detach()) for x in (total, mse, kl)]) * len(batch['labels'])

        metrics, prediction, weights = evaluate(model, valid, name, device)
        if best is None or selection(metrics, epoch) > selection(best['metrics'], best['epoch']):
            best = {'epoch': epoch, 'metrics': metrics}
            save_checkpoint(selected_path, {'model': model.state_dict(), **best})
            save_predictions(folder / 'valid_predictions.npz', valid, prediction, weights)
        loss, mse, kl = map(float, sums / len(train))
        history.append({
            'epoch': epoch, 'valid': metrics, 'train_loss': loss, 'main_MSE': mse,
            'KL': kl, 'seconds': time.monotonic() - started, 'best_epoch': best['epoch'],
        })
        save_checkpoint(resume_path, {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'history': history, 'epoch': epoch,
            'best': best, 'loader_rng': generator.get_state(), 'rng': rng_state(device),
        })
        save_json(folder / 'history.json', history)
        print(f'{name} {backbone} {arm} seed={seed} epoch={epoch} '
              f'valid_acc2={metrics["acc2"]:.4f} valid_f1={metrics["f1_weighted"]:.4f}', flush=True)

    selected = torch.load(selected_path, map_location='cpu')
    model.load_state_dict(selected['model'], strict=True)
    verified_metrics, _, _ = evaluate(model, valid, name, device)
    for metric in METRICS:
        np.testing.assert_allclose(verified_metrics[metric], selected['metrics'][metric], rtol=0, atol=1e-12)
    save_json(finished_path, {
        'best': best, 'trained_epochs': len(history), 'best_sha256': sha256(selected_path),
        'early_stopping': recovery_decision(history),
    })


def freeze(output, backbones, seeds):
    frozen = {}
    for backbone, seed, arm in itertools.product(backbones, seeds, ARMS):
        folder = model_folder(output, backbone, arm, seed)
        result = read_json(folder / 'complete.json')
        if sha256(folder / 'best.pt') != result['best_sha256']:
            raise ValueError(f'Checkpoint digest changed: {folder / "best.pt"}')
        frozen[f'{backbone}/{arm}/{seed}'] = result
    path = output / 'frozen_selections_before_TEST.json'
    if path.is_file() and read_json(path) != frozen:
        raise ValueError(f'Frozen selection changed: {path}')
    save_json(path, frozen)
    return frozen


def report(output, name, backbones, seeds, rows):
    groups = {}
    for backbone in backbones:
        groups[backbone] = {}
        for arm in ARMS:
            selected = [row['metrics'] for row in rows if row['backbone'] == backbone and row['arm'] == arm]
            groups[backbone][arm] = {
                'mean': {metric: mean(row[metric] for row in selected) for metric in METRICS},
                'sample_std': {
                    metric: stdev(row[metric] for row in selected) if len(selected) > 1 else None
                    for metric in METRICS
                },
            }
    keys = ('backbone', 'dataset', 'arm', 'seed', 'best_epoch')
    pending = output / 'results.pending.csv'
    with pending.open('w', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=[*keys, *METRICS])
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in keys}
            record.update({metric: row['metrics'][metric] for metric in METRICS})
            writer.writerow(record)
    replace_atomic(pending, output / 'results.csv')
    save_json(output / 'summary.json', {'dataset': name, 'seeds': seeds, 'backbones': groups, 'rows': rows})


def finish(output, cache, name, backbones, seeds, metadata, device):
    frozen = freeze(output, backbones, seeds)
    test = SequenceDataset(cache / 'test')
    rows = []
    for backbone, seed, arm in itertools.product(backbones, seeds, ARMS):
        folder = model_folder(output, backbone, arm, seed)
        checkpoint = frozen[f'{backbone}/{arm}/{seed}']
        prediction_path = folder / 'test_predictions.npz'
        evaluation_path = folder / 'test_evaluation.json'
        if evaluation_path.is_file():
            evaluation = read_json(evaluation_path)
            if (evaluation['checkpoint_sha256'] != checkpoint['best_sha256']
                    or evaluation['predictions_sha256'] != sha256(prediction_path)):
                raise ValueError(f'TEST artifacts differ from the recorded evaluation: {folder}')
        else:
            seed_all(seed)
            model = TransferModel(backbone, metadata['dims'], arm).to(device)
            state = torch.load(folder / 'best.pt', map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            del state
            metrics, prediction, weights = evaluate(model, test, name, device)
            save_predictions(prediction_path, test, prediction, weights)
            evaluation = {
                'backbone': backbone, 'dataset': name, 'arm': arm, 'seed': seed,
                'best_epoch': checkpoint['best']['epoch'], 'metrics': metrics,
                'checkpoint_sha256': checkpoint['best_sha256'],
                'predictions_sha256': sha256(prediction_path),
            }
            save_json(evaluation_path, evaluation)
            del model
        rows.append(evaluation)
    report(output, name, backbones, seeds, rows)
    print(f'{name}: wrote {len(rows)} TEST rows to {output / "summary.json"}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate the paired PI-LDW backbone adaptations in Table VI.')
    parser.add_argument('--config', type=Path, required=True, help='Dataset JSON used by pi_ldw.cli')
    parser.add_argument('--encoder-checkpoint', type=Path, required=True, help='From pretraining: encoders/T/best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbones', nargs='+', choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'pipeline'), default='pipeline')
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.config)
    name = config['dataset']
    if name not in DATASETS:
        raise ValueError(f'Table VI datasets are {", ".join(DATASETS)}; got {name}')
    features = Path(config['features']).expanduser().resolve(strict=True)
    annotations = Path(config['annotations']).expanduser().resolve(strict=True) if config.get('annotations') else None
    if name == 'unic' and annotations is None:
        raise ValueError('UniC requires the annotations path in its dataset configuration')
    bert = Path(config['bert']).expanduser().resolve(strict=True)
    checkpoint = args.encoder_checkpoint.expanduser().resolve(strict=True)
    device = torch.device(config.get('device', 'cuda:0'))
    seeds = list(args.seeds or config.get('seeds') or SEEDS)
    if len(set(seeds)) != len(seeds) or len(set(args.backbones)) != len(args.backbones):
        raise ValueError('Select each seed and backbone once')

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache' / name
    protocol = {
        'dataset': name, 'features': str(features), 'annotations': str(annotations) if annotations else None,
        'bert': str(bert), 'encoder_checkpoint': str(checkpoint), 'encoder_sha256': sha256(checkpoint),
        'device': str(device), 'backbones': args.backbones, 'seeds': seeds, 'arms': list(ARMS),
        'training': {
            'max_epochs': MAX_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY, 'kl_coefficient': KL_COEFFICIENT,
            'warmup_epochs': 1, 'gradient_clip': 1., 'early_stopping': POLICY,
        },
    }
    lock = output / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as active:
        active.write(str(os.getpid()))
    try:
        path = output / 'protocol.json'
        if path.is_file() and read_json(path) != protocol:
            raise ValueError(f'Configuration changed. Choose a new output directory: {output}')
        save_json(path, protocol)
        seed_all(1)
        if args.stage in ('prepare', 'pipeline'):
            metadata = prepare_cache(cache, name, features, annotations, bert, checkpoint, device)
        else:
            metadata = load_cache(cache)
        if args.stage in ('train', 'pipeline'):
            for backbone, seed, arm in itertools.product(args.backbones, seeds, ARMS):
                train_one(output, cache, name, backbone, arm, seed, metadata, device)
        if args.stage in ('evaluate', 'pipeline'):
            finish(output, cache, name, args.backbones, seeds, metadata, device)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
