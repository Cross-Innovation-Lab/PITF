from pathlib import Path
import hashlib,importlib,json,os,random,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
DATA={}
SETTINGS={}
def install(settings):
    SETTINGS.clear();SETTINGS.update(settings)
    DATA[settings['dataset']]=settings['features']
def configure(output,dataset='sims',seed=1):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'third_party/ConFEDE'/('SIMS' if dataset in ('sims','simsv2','unic') else dataset.upper())
    if 'config' in sys.modules and Path(sys.modules['config'].__file__).resolve()!=source/'config.py':
        raise RuntimeError('Run different dataset architectures in separate Python processes')
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    previous=Path.cwd()
    try:
        os.chdir(output);cfg=importlib.import_module('config')
    finally:os.chdir(previous)
    cfg.BERT_PATH=SETTINGS['bert'];cfg.seed=seed;cfg.DEVICE=torch.device(SETTINGS['device'])
    settings=getattr(cfg,'SIMS' if dataset in ('sims','simsv2','unic') else dataset.upper())
    cfg.DATASET_SETTINGS=settings
    if dataset=='unic':cfg.UNIC=settings
    settings.path.raw_data_path=DATA[dataset]
    settings.path.encoder_path=str(output/'encoders')+'/'
    settings.path.model_path=str(output/'fusion')+'/'
    from triagedmsa.mmap_pickle import load_mmap_pickle
    arrays=load_mmap_pickle(DATA[dataset])['train'];d=settings.downStream
    d.audio_seq_len,d.audio_fea_dim=arrays['audio'].shape[1:]
    d.video_seq_len,d.vision_fea_dim=arrays['vision'].shape[1:]
    if dataset=='unic':d.audio_dim_feedforward,d.vision_dim_feedforward=d.audio_fea_dim,d.vision_fea_dim
    return cfg
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_flash_sdp(False)
    torch.set_num_threads(4)
def replace_atomic(source,target):
    for attempt in range(20):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace_atomic(pending,path)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
