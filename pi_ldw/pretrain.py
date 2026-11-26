from __future__ import annotations
from pathlib import Path
import os,random,time,gc,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from reproductions.confede.runtime import DATA,configure,seed_all,save_json,sha256,replace_atomic
from reproductions.confede.data import SIMSData,slice_batch
from reproductions.confede.metrics import sentiment_metrics,conflict_metrics
from reproductions.confede.modules import predict_full,encode
from reproductions.confede.early_stopping import mono_stop_decision
def amp(args):
    return torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.precision=='bf16')

def build_model(stage,cfg):
    from model.net.constrastive.text_encoder_finetune import TextPretrain
    from model.net.constrastive.audio_encoder_fintune import AudioPretrain
    from model.net.constrastive.vision_encoder_finetune import VisionPretrain
    from model.net.constrastive.TVA_fusion import TVA_fusion
    return {'T':TextPretrain,'A':AudioPretrain,'V':VisionPretrain,'fusion':TVA_fusion}[stage](config=cfg).to(cfg.DEVICE)

def mono_forward(model,stage,batch,device):
    label=batch['regression_labels_'+stage].to(device)
    if stage=='T':
        return model(batch['raw_text'],label,return_loss=True)
    key={'A':'audio','V':'vision'}[stage]
    return model(batch[key].to(device),label,batch[key+'_padding_mask'].to(device),return_loss=True)

def evaluate(model,stage,ds,cfg,args,batch_size=16):
    model.eval()
    modes='MTAV' if stage=='fusion' else stage
    predictions={m:[] for m in modes}
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            if stage=='fusion':
                clean={k:v for k,v in batch.items() if not k.startswith('regression_labels')}
                out=predict_full(model,clean,cfg.DEVICE)
            else:
                if stage=='T':
                    value=model.encoder(batch['raw_text'],device=cfg.DEVICE)
                else:
                    key={'A':'audio','V':'vision'}[stage]
                    value=model.encoder(batch[key].to(cfg.DEVICE),batch[key+'_padding_mask'].to(cfg.DEVICE),device=cfg.DEVICE)
                out={stage:model.classifier(value).reshape(-1)}
            for m in modes:predictions[m].append(out[m].float().cpu().numpy())
    predictions={m:np.concatenate(parts) for m,parts in predictions.items()}
    metrics={m:sentiment_metrics(ds.labels[m],predictions[m],args.dataset) for m in modes}
    if stage=='fusion' and args.dataset in ('sims','simsv2'):metrics['conflict']=conflict_metrics(ds.labels,predictions)
    return metrics,predictions

def checkpoint(path,model,optimizer,scheduler,epoch,best,args):
    pending=path.with_suffix('.pending')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'best':best,'args':vars(args),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}},pending)
    replace_atomic(pending,path)

def update_neighbors(model,ds,cfg):
    model.eval();features=[]
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
            features.append(torch.cat(encode(model,batch,cfg.DEVICE),dim=-1).float().cpu())
    ds.update_neighbors(torch.cat(features))

def run_training(args,stage):
    output=Path(args.output_dir).resolve();cfg=configure(output,args.dataset,args.seed)
    seed_all(args.seed)
    folder=output/('encoders/'+stage if stage!='fusion' else f'fusion/seed_{args.seed}')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print(f'SKIP completed {folder}',flush=True);return
    if (folder/'last.pt').exists() and not args.resume:
        raise FileExistsError(f'Use --resume to continue {folder}')
    train=SIMSData(DATA[args.dataset],'train',sampler_mode=getattr(args,'sampler_mode','corrected'),dataset=args.dataset);valid=SIMSData(DATA[args.dataset],'valid')
    model=build_model(stage,cfg)
    ds_cfg=cfg.DATASET_SETTINGS.downStream
    stage_cfg=getattr(ds_cfg,{'T':'textPretrain','A':'audioPretrain','V':'visionPretrain','fusion':'TVAExp_fusion'}[stage])
    epochs=args.epochs or stage_cfg.epoch
    logical_batch=stage_cfg.batch_size
    stage_micro=getattr(args,'fusion_micro_batch' if stage=='fusion' else 'mono_micro_batch',None)
    micro=args.micro_batch or stage_micro or (16 if stage=='fusion' and args.dataset=='sims' else 2 if stage=='fusion' else 16)
    lr=stage_cfg.lr
    warmup=stage_cfg.num_warm_up
    start_save=(epochs//3 if stage=='T' else (10 if args.dataset in ('sims','simsv2') else 1) if stage=='fusion' else 5 if args.dataset=='mosei' else 50 if stage=='V' and args.dataset in ('sims','simsv2') else 20 if stage=='V' else 30)
    start_save=min(start_save,max(0,epochs-1))
    loader=DataLoader(train,batch_size=logical_batch,shuffle=True,num_workers=0)
    if stage=='fusion':
        model.set_train([False,False,True,True])
        for mode,name in [('T','text'),('V','vision'),('A','audio')]:
            encoder_source=Path(getattr(args,'encoder_source_dir',None) or output/'encoders')
            payload=torch.load(encoder_source/mode/'best.pt',map_location='cpu')
            state={k[len('encoder.'):]:v for k,v in payload['model'].items() if k.startswith('encoder.')}
            getattr(model,name+'_encoder').load_state_dict(state,strict=True)
            del payload,state
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in ['bias','LayerNorm.weight'])],'weight_decay':1e-3},
            {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in ['bias','LayerNorm.weight'])],'weight_decay':0.0}],lr=lr)
    elif stage=='T':
        from transformers.optimization import AdamW
        optimizer=AdamW(model.parameters(),lr=lr,weight_decay=1e-3,no_deprecation_warning=True)
    elif args.dataset=='mosei':
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in ['bias','LayerNorm.weight'])],'weight_decay':stage_cfg.decay},
            {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in ['bias','LayerNorm.weight'])],'weight_decay':0.0}],lr=lr)
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-3,amsgrad=True)
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup*len(loader),epochs*len(loader))
    start=0;best={'selection':None,'epoch':None}
    history=[]
    if args.resume and (folder/'last.pt').exists():
        previous=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        expected={'dataset':args.dataset,'seed':args.seed,'stage':stage,'epochs':epochs,
                  'logical_batch':logical_batch,'micro_batch':micro,'precision':args.precision}
        for key,value in expected.items():
            if previous[key]!=value:raise ValueError(f'Resume protocol mismatch for {key}: {previous[key]} != {value}')
        if previous.get('sampler_mode','corrected') != getattr(args,'sampler_mode','corrected'):
            raise ValueError('Resume sampler mismatch')
        saved=torch.load(folder/'last.pt',map_location=cfg.DEVICE)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['epoch'];best=saved['best']
        random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']])
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))[:start]
        if stage=='fusion' and (folder/'neighbor_pools.json').exists():
            train.pools=json.loads((folder/'neighbor_pools.json').read_text(encoding='utf-8'))
        del saved
    protocol={'dataset':args.dataset,'data':DATA[args.dataset],'seed':args.seed,'stage':stage,'epochs':epochs,
              'logical_batch':logical_batch,'micro_batch':micro,'precision':args.precision,
              'learning_rate':lr,'warmup_epochs':warmup,'save_after_epoch':start_save,
              'selection':'VALID Acc2' if stage=='fusion' and args.dataset in ('sims','simsv2') else 'VALID MAE',
              'early_stopping':{'enabled':stage!='fusion' and args.mono_early_stop_patience>0,
                   'patience_epochs':args.mono_early_stop_patience if stage!='fusion' else 0,
                   'minimum_epochs':max(args.mono_early_stop_min_epochs,start_save+1),
                   'monitor':'VALID MAE','learning_rate_schedule_horizon_unchanged':True},
              'sampler_mode':getattr(args,'sampler_mode','corrected'),
              'test_used_for_training':False,'upstream_commit':'640d8831b6c8befe2a438fe59671caee25ec4edd'}
    save_json(folder/'protocol.json',protocol)
    print('TRAIN_START '+json.dumps(protocol),flush=True)
    completed_epoch=start
    stop=mono_stop_decision(stage,start,best['epoch'],patience=args.mono_early_stop_patience,
                           min_epochs=args.mono_early_stop_min_epochs,save_after_epoch=start_save,
                           checkpoint_exists=(folder/'best.pt').exists())
    for epoch in range(start+1,epochs+1):
        if stop is not None:break
        begun=time.monotonic()
        interval=2 if args.dataset=='mosei' else 10
        if stage=='fusion' and (epoch%interval==1 or train.pools is None):
            print(f'NEIGHBORS epoch={epoch} TRAIN only',flush=True)
            update_neighbors(model,train,cfg);save_json(folder/'neighbor_pools.json',train.pools)
        if stage=='T':model.set_train([epoch>=epochs//3,True,True])
        elif stage in 'AV':model.set_train([True,True])
        model.train();loss_sum=0.;count=0
        for step,batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            batch_n=len(batch['index'])
            neighbors=train.sample_neighbors(batch['index']) if stage=='fusion' else None
            for left in range(0,batch_n,micro):
                right=min(left+micro,batch_n);small=slice_batch(batch,left,right)
                with amp(args):
                    if stage=='fusion':
                        neighbor=slice_batch(neighbors,6*left,6*right)
                        result=model(small,neighbor,return_loss=True)
                        loss=result[2]
                    else:loss=mono_forward(model,stage,small,cfg.DEVICE)[2]
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {stage} loss epoch={epoch}, step={step}')
                (loss*((right-left)/batch_n)).backward()
                loss_sum+=float(loss.detach())*(right-left);count+=right-left
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            if stage=='fusion' and args.dataset=='mosei' and epoch>start_save and (step+1)%100==0:
                interim,interim_predictions=evaluate(model,stage,valid,cfg,args)
                if best['selection'] is None or interim['M']['mae']<best['selection']:
                    best={'selection':interim['M']['mae'],'epoch':epoch,'step':step+1,'metrics':interim}
                    pending=folder/'best.pending'
                    torch.save({'model':model.state_dict(),'epoch':epoch,'step':step+1,'metrics':interim,'protocol':protocol},pending)
                    replace_atomic(pending,folder/'best.pt')
                    np.savez_compressed(folder/'valid_predictions.npz',**interim_predictions,**{'truth_'+m:valid.labels[m] for m in interim_predictions})
                model.train()
        metrics,predictions=evaluate(model,stage,valid,cfg,args)
        maximize=stage=='fusion' and args.dataset in ('sims','simsv2')
        score=metrics['M' if stage=='fusion' else stage]['acc2' if maximize else 'mae']
        improved=best['selection'] is None or (score>best['selection'] if maximize else score<best['selection'])
        if epoch>start_save and improved and not(stage=='fusion' and args.dataset=='mosei'):
            best={'selection':score,'epoch':epoch,'metrics':metrics}
            pending=folder/'best.pending'
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'protocol':protocol},pending)
            replace_atomic(pending,folder/'best.pt')
            np.savez_compressed(folder/'valid_predictions.npz',**{m:predictions[m] for m in predictions},
                                **{'truth_'+m:valid.labels[m] for m in predictions})
        row={'epoch':epoch,'train_loss':loss_sum/count,'valid':metrics,'seconds':time.monotonic()-begun,'best_epoch':best['epoch']}
        history.append(row);save_json(folder/'history.json',history)
        checkpoint(folder/'last.pt',model,optimizer,scheduler,epoch,best,args)
        brief=metrics['M' if stage=='fusion' else stage]
        print(f'EPOCH stage={stage} seed={args.seed} epoch={epoch}/{epochs} loss={row["train_loss"]:.5f} valid_acc2={brief["acc2"]:.4f} valid_f1={brief["f1_weighted"]:.4f} valid_mae={brief["mae"]:.4f} best={best["epoch"]} seconds={row["seconds"]:.1f}',flush=True)
        save_json(output/'status.json',{'state':'training','stage':stage,'seed':args.seed,'epoch':epoch,'epochs':epochs,'best':best,'latest':row})
        pause_path=getattr(args,'pause_after_epoch_path',None)
        if pause_path and Path(pause_path).exists():
            from pi_ldw.cli import TrainingPaused
            save_json(output/'status.json',{'state':'paused','stage':stage,'seed':args.seed,'epoch':epoch,
                      'checkpoint':str(folder/'last.pt'),'resume_requires_user_authorization':True})
            raise TrainingPaused()
        completed_epoch=epoch
        stop=mono_stop_decision(stage,epoch,best['epoch'],patience=args.mono_early_stop_patience,
                               min_epochs=args.mono_early_stop_min_epochs,save_after_epoch=start_save,
                               checkpoint_exists=(folder/'best.pt').exists())
    if stop is not None:
        save_json(folder/'early_stop.json',stop)
        print('EARLY_STOP '+stage+' '+json.dumps(stop),flush=True)
    save_json(folder/'complete.json',{'state':'completed','best':best,'epochs':epochs,
              'trained_epochs':completed_epoch,'stopped_early':stop is not None,
              'early_stop':stop,'protocol':protocol})
    (folder/'last.pt').unlink(missing_ok=True)
    del model,optimizer,scheduler,train,valid;gc.collect();torch.cuda.empty_cache()
