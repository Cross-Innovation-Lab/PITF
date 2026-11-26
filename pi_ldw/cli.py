from pathlib import Path
import argparse
import gc
import json
import os
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from reproductions.confede import runtime as rt
from reproductions.confede.data import SIMSData,slice_batch
from reproductions.confede.metrics import sentiment_metrics
from reproductions.confede.modules import encode
from reproductions.confede_ldw.intensity_contrast import IntensityContrastJoint
from reproductions.confede_ldw.difference_geometry import DifferenceGeometryJoint, mean_entropy, fit_entropy_temperature
from reproductions.confede_ldw.component_ablation import ComponentAblationJoint, ComponentAblationV2
from reproductions.confede_ldw.mosi_no_cl import MOSINoCLFusion
from reproductions.confede_ldw.pitf import GEOMETRIES, PITFFusion, teacher_distances
from reproductions.confede_ldw.pitf_factorial import ARMS as FACTORIAL_ARMS, build_factorial_pitf
from reproductions.confede_ldw.sims_no_ldw import SIMSNoLDWFusion
from reproductions.confede_ldw.self_stopping import recovery_decision
from . import pretrain as pre

ARMS = ('no_ldw', *GEOMETRIES, *FACTORIAL_ARMS)
ENGLISH_DATASETS = ('mosi', 'mosei')
OPTIONS={}
class TrainingPaused(InterruptedError):pass
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def free():gc.collect();torch.cuda.empty_cache()
def save_torch(path,value):
    path=Path(path);pending=path.with_suffix('.pending');torch.save(value,pending);rt.replace_atomic(pending,path)
def state_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])
def dataset(path, split, sampler_mode='upstream', dataset=None, **kwargs):
    name = dataset or OPTIONS['dataset']
    if name == 'unic':
        from reproductions.confede_ldw.unic_data import UniCData
        return UniCData(path, split, sampler_mode=sampler_mode, annotations=OPTIONS['annotations'], **kwargs)
    data = SIMSData(path, split, sampler_mode=sampler_mode, dataset=name, **kwargs)
    if name not in ENGLISH_DATASETS:
        required = ('regression_labels_' + mode for mode in 'TAV')
        missing = [key for key in required if key not in data.data]
        if missing:
            raise ValueError(f'{name} {split} is missing unimodal labels: {", ".join(missing)}')
    else:
        data.labels.update({mode: data.labels['M'].copy() for mode in 'TAV'})
    return data


class EnglishData(SIMSData):
    def __init__(self, path, split, teacher_bank=None):
        super().__init__(path, split, dataset=OPTIONS['dataset'], sampler_mode='upstream')
        self.raw_M = self.labels['M'].copy()
        normalized = self.raw_M / np.float32(3)
        self.labels = {mode: normalized.copy() for mode in 'MTAV'}
        if teacher_bank is not None:
            self.labels.update(zip('TAV', teacher_bank.T))

    def update_neighbors(self, embeddings):
        normalized = self.labels['M']
        self.labels['M'] = self.raw_M
        try:
            super().update_neighbors(embeddings)
        finally:
            self.labels['M'] = normalized


def fusion_data(split, teacher_bank=None):
    if OPTIONS['dataset'] in ENGLISH_DATASETS:
        return EnglishData(OPTIONS['features'], split, teacher_bank)
    return dataset(OPTIONS['features'], split)


def encoder_dir(output):
    return Path(OPTIONS['encoders']) if OPTIONS.get('encoders') else output / 'encoders'
def key(metrics,epoch,seed):
    reference=OPTIONS.get('selection_reference')
    baseline=reference.get(str(seed),reference) if reference else {}
    return (min(metrics[k]-baseline.get(k,0.) for k in ('acc2','f1_weighted')),
        metrics['acc2']+metrics['f1_weighted'],-metrics['mae'],-epoch)
def optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if not any(s in n for s in ('bias','LayerNorm.weight'))],weight_decay=1e-3),
        dict(params=[p for n,p in model.named_parameters() if any(s in n for s in ('bias','LayerNorm.weight'))],weight_decay=0.)],lr=OPTIONS['lr'])
def build(output, arm, seed):
    cfg = rt.configure(output / 'runtime' / arm / f'seed_{seed}', OPTIONS['dataset'], seed)
    rt.seed_all(seed)
    base = pre.build_model('fusion', cfg)
    for mode, name in (('T', 'text'), ('A', 'audio'), ('V', 'vision')):
        checkpoint = torch.load(encoder_dir(output) / mode / 'best.pt', map_location='cpu')
        state = {key[8:]: value for key, value in checkpoint['model'].items()
                 if key.startswith('encoder.')}
        getattr(base, f'{name}_encoder').load_state_dict(state, strict=True)
        del checkpoint, state

    calibration = read(output / 'calibration.json')['arms']
    geometry = 'polarity_intensity' if arm in FACTORIAL_ARMS else arm
    temperature = calibration.get(geometry, {}).get('temperature', 1.0)
    if OPTIONS['architecture'] == 'pitf':
        original = IntensityContrastJoint(base, contrastive_mode='off')
        if arm in FACTORIAL_ARMS or arm == 'no_ldw':
            model = build_factorial_pitf(original, temperature, '000' if arm == 'no_ldw' else arm)
        else:
            model = PITFFusion(original, temperature, geometry=geometry)
    elif OPTIONS['dataset'] in ENGLISH_DATASETS or OPTIONS['dataset'] == 'simsv2':
        with np.load(output / 'teacher_train.npz') as data:
            bank = data['TAV']
        base_arm = 'absolute' if arm in ('polarity_only', 'intensity_only') else arm
        model = MOSINoCLFusion(base, base_arm, bank, temperature)
        if base_arm != arm:
            model = ComponentAblationV2(model, arm, temperature)
    else:
        original = IntensityContrastJoint(base, contrastive_mode='off')
        model = original
        if arm == 'no_ldw':
            model = SIMSNoLDWFusion(original)
        elif arm == 'polarity_intensity':
            model = DifferenceGeometryJoint(original, arm, temperature=temperature)
        elif arm in ('polarity_only', 'intensity_only'):
            model = ComponentAblationJoint(original, arm, temperature)
    model.set_train([False, False, True, True])
    return cfg, model.to(cfg.DEVICE)
@torch.no_grad()
def evaluate(model,data,device):
    model.eval();pred={m:[] for m in 'MTAV'}
    for batch in DataLoader(data,batch_size=OPTIONS['eval_batch'],shuffle=False,num_workers=0):
        clean={k:v for k,v in batch.items() if not k.startswith('regression_labels') and k!='index'}
        values=model.infer(clean,device)
        for m in pred:pred[m].append(values[m].float().cpu().numpy())
    pred={m:np.concatenate(v) for m,v in pred.items()}
    english=OPTIONS['dataset'] in ('mosi','mosei')
    if english:pred={m:v*np.float32(3) for m,v in pred.items()}
    truth={m:data.raw_M if english else data.labels[m] for m in pred}
    return {m:sentiment_metrics(truth[m],v,OPTIONS['dataset']) for m,v in pred.items()},pred
@torch.no_grad()
def neighbors(model,data,cfg):
    model.eval();parts=[]
    for batch in DataLoader(data,batch_size=OPTIONS['neighbor_batch'],shuffle=False,num_workers=0):
        parts.append(torch.cat(encode(model,batch,cfg.DEVICE),-1).float().cpu())
    data.update_neighbors(torch.cat(parts))
def pretrain(output):
    if OPTIONS.get('encoders'):
        for mode in 'TAV':
            path = encoder_dir(output) / mode / 'best.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
        return
    pre.SIMSData=dataset
    for mode in 'TAV':
        if (output/'PAUSE').exists():raise TrainingPaused()
        args=argparse.Namespace(output_dir=str(output),dataset=OPTIONS['dataset'],seed=OPTIONS['encoder_seed'],
            epochs=OPTIONS['encoder_epochs'][mode],micro_batch=None,mono_micro_batch=OPTIONS['mono_micro'][mode],
            fusion_micro_batch=OPTIONS['micro_batch'],precision='fp32',resume=True,sampler_mode='upstream',
            mono_early_stop_patience=OPTIONS['mono_patience'],mono_early_stop_min_epochs=OPTIONS['mono_min_epochs'],
            pause_after_epoch_path=None)
        pre.run_training(args,mode)
def prepare(output):
    if (output / 'calibration.json').exists():
        return
    train = fusion_data('train')
    english = OPTIONS['dataset'] in ENGLISH_DATASETS
    if english:
        raw = dataset(OPTIONS['features'], 'train')
        cfg = rt.configure(output / 'teacher_runtime', OPTIONS['dataset'], OPTIONS['encoder_seed'])
        predictions = []
        for mode in 'TAV':
            model = pre.build_model(mode, cfg)
            checkpoint = torch.load(encoder_dir(output) / mode / 'best.pt', map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=True)
            args = argparse.Namespace(dataset=OPTIONS['dataset'])
            _, values = pre.evaluate(model, mode, raw, cfg, args, batch_size=OPTIONS['eval_batch'])
            predictions.append(values[mode])
            del model, checkpoint
            free()
        bank = np.clip(np.stack(predictions, axis=1), -3, 3) / np.float32(3)
    else:
        bank = np.stack([train.labels[mode] for mode in 'TAV'], axis=1).astype(np.float32)
    np.savez_compressed(output / 'teacher_train.npz', TAV=bank, indices=np.arange(len(train)))

    unimodal = torch.as_tensor(bank, dtype=torch.float64)
    multimodal = torch.as_tensor(train.labels['M'], dtype=torch.float32).double()
    target = mean_entropy(teacher_distances(unimodal, multimodal, 'absolute'), 1.0)
    arms = {'absolute': dict(temperature=1.0, matched_entropy=target)}
    requested = set(OPTIONS['arms'])
    if any(arm in requested for arm in ('100', '101', '110', '111')):
        requested.add('polarity_intensity')
    for geometry in GEOMETRIES:
        if geometry not in requested or geometry == 'absolute':
            continue
        values = teacher_distances(unimodal, multimodal, geometry)
        temperature = fit_entropy_temperature(values, target)
        arms[geometry] = dict(temperature=temperature, matched_entropy=mean_entropy(values, temperature))
    rt.save_json(output / 'calibration.json', dict(split='TRAIN', n=len(train), target_entropy=target,
                 arms=arms, label_source='predicted' if english else 'real'))
def train_one(output,arm,seed):
    folder=output/'arms'/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    cfg, model = build(output, arm, seed)
    with np.load(output / 'teacher_train.npz') as teachers:
        train = fusion_data('train', teachers['TAV'])
    valid = fusion_data('valid')
    loader=DataLoader(train,batch_size=OPTIONS['logical_batch'],shuffle=True,num_workers=0)
    opt=optimizer(model);schedule=get_linear_schedule_with_warmup(opt,len(loader),OPTIONS['max_epochs']*len(loader))
    init=state_hash(model.text_encoder);history=[];best=None;start=0
    if (folder/'last.pt').exists():
        previous=torch.load(folder/'last.pt',map_location='cpu');model.load_state_dict(previous['model'])
        opt.load_state_dict(previous['optimizer']);schedule.load_state_dict(previous['scheduler'])
        history,best,start=previous['history'],previous['best'],previous['epoch'];train.pools=previous['pools'];restore(previous['rng']);del previous
    for epoch in range(start+1,OPTIONS['max_epochs']+1):
        if recovery_decision(history)['stop']:break
        if (output/'PAUSE').exists():raise TrainingPaused()
        begun=time.monotonic()
        if (epoch - 1) % OPTIONS['neighbor_refresh'] == 0 or train.pools is None:neighbors(model,train,cfg)
        model.train();loss_sum=0.;count=0
        for step,batch in enumerate(loader,1):
            opt.zero_grad(set_to_none=True);n=len(batch['index']);partner=train.sample_neighbors(batch['index'])
            for left in range(0,n,OPTIONS['micro_batch']):
                right=min(left+OPTIONS['micro_batch'],n)
                result=model(slice_batch(batch,left,right),slice_batch(partner,6*left,6*right),return_loss=True)
                loss=result[2];assert torch.isfinite(loss)
                (loss*((right-left)/n)).backward();loss_sum+=float(loss.detach())*(right-left);count+=right-left
                del result,loss
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();schedule.step()
            if step==1 or step%10==0 or step==len(loader):
                rt.save_json(output/'status.json',dict(state='training',arm=arm,seed=seed,epoch=epoch,step=step,steps=len(loader),train_loss=loss_sum/count))
        metrics,predictions=evaluate(model,valid,cfg.DEVICE)
        if best is None or key(metrics['M'],epoch,seed)>key(best['metrics'],best['epoch'],seed):
            best=dict(epoch=epoch,metrics=metrics['M'])
            save_torch(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,metrics=metrics))
            np.savez_compressed(folder/'valid_predictions.npz',**predictions,ids=np.asarray(valid.data['id']).astype(str))
        assert state_hash(model.text_encoder)==init
        history.append(dict(epoch=epoch,valid=metrics,train_loss=loss_sum/count,seconds=time.monotonic()-begun,best_epoch=best['epoch']))
        save_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
            epoch=epoch,history=history,best=best,pools=train.pools,rng=rng()))
        rt.save_json(folder/'history.json',history)
        print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best=best,stop=recovery_decision(history))),flush=True)
    payload=torch.load(folder/'best.pt',map_location='cpu');model.load_state_dict(payload['model'],strict=True)
    replay,_=evaluate(model,valid,cfg.DEVICE)
    for k,v in payload['metrics']['M'].items():
        if isinstance(v,(int,float)):np.testing.assert_allclose(replay['M'][k],v,atol=1e-12,rtol=0)
    rt.save_json(folder/'complete.json',dict(best=best,trained_epochs=len(history),best_sha256=rt.sha256(folder/'best.pt'),
        frozen_text_unchanged=True,best_VALID_replay_passed=True,early_stopping=recovery_decision(history)))
    del model,opt,schedule,train,valid;free()
def finish(output):
    frozen={}
    for arm in OPTIONS['arms']:
        for seed in OPTIONS['seeds']:
            f=output/'arms'/arm/f'seed_{seed}';c=read(f/'complete.json')
            assert rt.sha256(f/'best.pt')==c['best_sha256'];frozen[f'{arm}/{seed}']=c
    target=output/'frozen_selections_before_TEST.json'
    if target.exists():assert read(target)==frozen
    else:rt.save_json(target,frozen)
    test=fusion_data('test');rows=[];english=OPTIONS['dataset'] in ('mosi','mosei')
    for arm in OPTIONS['arms']:
        for seed in OPTIONS['seeds']:
            if (output/'PAUSE').exists():raise TrainingPaused()
            f=output/'arms'/arm/f'seed_{seed}'
            if not (f/'test_evaluation.json').exists():
                cfg,model=build(output,arm,seed);saved=torch.load(f/'best.pt',map_location='cpu')
                model.load_state_dict(saved['model'],strict=True);del saved
                metrics,pred=evaluate(model,test,cfg.DEVICE)
                np.savez_compressed(f/'test_predictions.npz',**pred,ids=np.asarray(test.data['id']).astype(str),
                    **{'truth_'+m:test.raw_M if english else test.labels[m] for m in 'MTAV'})
                rt.save_json(f/'test_evaluation.json',dict(seed=seed,arm=arm,split='TEST',metrics=metrics,checkpoint=frozen[f'{arm}/{seed}']))
                del model;free()
            rows.append(read(f/'test_evaluation.json'))
    groups={}
    for arm in OPTIONS['arms']:
        items=[r for r in rows if r['arm']==arm];keys=[k for k,v in items[0]['metrics']['M'].items() if isinstance(v,(int,float)) and k!='n']
        groups[arm]=dict(rows=items,mean={k:float(np.mean([r['metrics']['M'][k] for r in items])) for k in keys},
            sample_std={k:float(np.std([r['metrics']['M'][k] for r in items],ddof=1)) if len(items)>1 else None for k in keys})
    rt.save_json(output/'summary.json',dict(dataset=OPTIONS['dataset'], architecture=OPTIONS['architecture'],
                                          split='TEST',seeds=OPTIONS['seeds'],groups=groups))
    rt.save_json(output/'status.json',dict(state='completed',models=len(rows)))
def main():
    p=argparse.ArgumentParser(description='Train and evaluate PITF, its ablations, and the ConFEDE controls.')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('check-data','pretrain','prepare','train','evaluate','pipeline'),default='pipeline')
    p.add_argument('--arms', nargs='+', choices=ARMS)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--architecture', choices=('pitf', 'confede'), default='pitf')
    p.add_argument('--encoders', help='Directory containing T/best.pt, A/best.pt and V/best.pt')
    p.add_argument('--device', help='PyTorch device, for example cuda:0 or cpu')
    args=p.parse_args();settings=read(args.config)
    settings['architecture'] = args.architecture
    settings['encoders'] = str(Path(args.encoders).resolve()) if args.encoders else None
    if args.device:
        settings['device'] = args.device
    for k in ('features','annotations','bert'):
        if settings.get(k):settings[k]=str(Path(settings[k]).expanduser().resolve())
    if args.arms:settings['arms']=args.arms
    if args.seeds:settings['seeds']=args.seeds
    if not settings['seeds'] or len(set(settings['seeds']))!=len(settings['seeds']):raise ValueError('Unique nonempty seeds required')
    if not settings['arms'] or len(set(settings['arms'])) != len(settings['arms']) or not set(settings['arms']) <= set(ARMS):
        raise ValueError('Expected unique experiment arms from --help')
    if args.architecture == 'confede' and set(settings['arms']) & set(FACTORIAL_ARMS):
        raise ValueError('Binary component combinations require --architecture pitf')
    if isinstance(settings.get('selection_reference'),str):settings['selection_reference']=read(settings['selection_reference'])
    OPTIONS.clear()
    OPTIONS.update(settings)
    rt.install(settings)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.stage=='check-data':
        report={}
        for split in ('train','valid','test'):
            ds=dataset(settings['features'],split);ids=list(map(str,ds.data['id']))
            assert len(ids)==len(set(ids))==len(ds)
            for m in 'MTAV':assert np.isfinite(ds.labels[m]).all()
            report[split]=dict(n=len(ds),audio_shape=list(ds.data['audio'].shape),vision_shape=list(ds.data['vision'].shape))
        rt.save_json(output/'data_check.json',report);print(json.dumps(report));return
    lock = output / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as active:
        active.write(str(os.getpid()))
    try:
        protocol = output / 'protocol.json'
        if protocol.exists() and read(protocol) != settings:
            raise ValueError(f'Configuration changed. Use a new output directory: {output}')
        if not protocol.exists():
            rt.save_json(protocol, settings)
        try:
            if args.stage in ('pretrain', 'pipeline'):
                pretrain(output)
            if args.stage in ('prepare', 'pipeline'):
                prepare(output)
            if args.stage in ('train', 'pipeline'):
                for seed in settings['seeds']:
                    for arm in settings['arms']:
                        train_one(output, arm, seed)
            if args.stage in ('evaluate', 'pipeline'):
                finish(output)
        except TrainingPaused:
            rt.save_json(output / 'status.json', dict(state='paused'))
        except Exception as exc:
            rt.save_json(output / 'status.json', dict(state='failed', error=repr(exc)))
            raise
    finally:
        lock.unlink()
if __name__=='__main__':main()
