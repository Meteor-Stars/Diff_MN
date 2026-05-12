# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
import time

from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from utils_diffmn.cli_utils import nondefault_trainer_args
from utils_diffmn.callback_utils import prepare_trainer_configs
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from pathlib import Path
import datetime
from utils_diffmn.cli_utils import nondefault_trainer_args

# data_root = os.environ['DATA_ROOT']
data_root='./data'

def find_max_epoch_element(ckpt_list):
    max_epoch = -1
    max_element = None

    for ckpt in ckpt_list:
        if ckpt == "last.ckpt":
            continue

        epoch_str = ckpt.split('-')[0]
        epoch = int(epoch_str)

        if epoch > max_epoch:
            max_epoch = epoch
            max_element = ckpt

    return max_element,max_epoch


def init_model_data_trainer(parser):
    
    opt, unknown = parser.parse_known_args()
    opt.resume=False
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.name:
        name = opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size

    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"

    moe_num_experts = 4
    d_name = 'stock' #['sine','mujoco','energy','stock']
    opt.gpus = '4,'
    miss_=0.3 #0.5 0.7
    seq_len = 36 #12 24

    opt.medical_datasets=['ECG200','ECG5000','TwoLeadECG','ECGFiveDays']

    #ploymial
    # d_name='polynomial'
    # opt.baseline=False
    # opt.gpus = '0,'
    # seq_len=24
    # opt.gpus = '0,'
    # miss_=0.7

    #medical datasets #['ECG200','ECG5000','TwoLeadECG','ECGFiveDays']
    # opt.gpus = '0,'
    # d_name = 'ECG200' #ECG5000 140 #ECGFiveDays 136 TwoLeadECG 82
    # seq_len = 96
    # miss_=0.3
    # miss_ = 0.5
    miss_ = 0.7

    opt.baseline=False
    opt.d_name=d_name
    opt.miss_=miss_
    opt.seq_len=seq_len
    opt.dataset=d_name

    opt.normal_datasets=['sine','mujoco','energy','stock']

    opt.baseline=True

    opt.device='cuda:'+str(opt.gpus)[:1]

    nowname = 'dataset={}-miss={}seqlen={}'.format(d_name, miss_, seq_len)

    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False


    config.seq_length=opt.seq_len+moe_num_experts
    config.model.params.seq_len=config.seq_length
    config.model.params.unet_config.params.seq_len=config.seq_length
    config.model.params.cond_stage_config.params.window=config.seq_length


    config.lightning.trainer.max_epochs=600
    # config.lightning.trainer.max_epochs=2
    if opt.d_name=='sine':
        config.model.params.unet_config.params.in_channels=5
        config.model.params.unet_config.params.out_channels = 5
        config.model.params.channels = 5
        opt.inp_dim = 5
    elif opt.d_name=='stock':

        config.model.params.unet_config.params.in_channels=6
        config.model.params.unet_config.params.out_channels = 6
        config.model.params.channels = 6
        opt.inp_dim = 6
    elif opt.d_name=='energy':

        config.model.params.unet_config.params.in_channels=28
        config.model.params.unet_config.params.out_channels = 28
        config.model.params.channels = 28
        opt.inp_dim = 28
    elif opt.d_name=='mujoco':

        config.model.params.unet_config.params.in_channels=14
        config.model.params.unet_config.params.out_channels = 14
        config.model.params.channels=14
        opt.inp_dim=14
    elif opt.d_name=='polynomial':
        config.model.params.unet_config.params.in_channels=1
        config.model.params.unet_config.params.out_channels = 1
        config.model.params.channels=1
        opt.inp_dim=1

    elif opt.d_name=='ECG200':
        config.model.params.unet_config.params.in_channels=1
        config.model.params.unet_config.params.out_channels = 1
        config.model.params.channels=1
        opt.inp_dim=1
    elif opt.d_name=='ECG5000':
        config.model.params.unet_config.params.in_channels=1
        config.model.params.unet_config.params.out_channels = 1
        config.model.params.channels=1
        opt.inp_dim=1
    elif opt.d_name=='TwoLeadECG':
        config.model.params.unet_config.params.in_channels=1
        config.model.params.unet_config.params.out_channels = 1
        config.model.params.channels=1
        opt.inp_dim=1
    elif opt.d_name=='ECGFiveDays':
        config.model.params.unet_config.params.in_channels=1
        config.model.params.unet_config.params.out_channels = 1
        config.model.params.channels=1
        opt.inp_dim=1

    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    opt.logdir=logdir
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    opt.ckptdir=ckptdir
    metrics_dir = Path(logdir) / 'metric_dict.pkl'
    if metrics_dir.exists():
        print(f"Metric exists! Skipping {nowname}")
        sys.exit(0)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)

    lightning_config.trainer = trainer_config

    # model
    if opt.resume:
        ckpt_path = logdir +'/checkpoints' + '/last.ckpt'
        # print(os.listdir(logdir +'/checkpoints'))
        check_lists=os.listdir(logdir +'/checkpoints')
        max_element,max_epoch=find_max_epoch_element(check_lists)
        ckpt_path = logdir + '/checkpoints' +'/'+max_element

        config.model['params']['ckpt_path'] = ckpt_path


    model = instantiate_from_config(config.model)

    trainer_kwargs = prepare_trainer_configs(nowname, logdir, opt, lightning_config, ckptdir, model, now, cfgdir, config, trainer_opt)

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))

    config.data.params.args = vars(opt)
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    assert config.data.params.input_channels == 1, \
        "Assertion failed: Only univariate input is supported. Please ensure input_channels == 1."
    print("#### Data Preparation Finished #####")

    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        
    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; # type: ignore
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    
    return model, data, trainer, opt, logdir, melk


def load_model_data(parser):
    
    opt, unknown = parser.parse_known_args()
        
    if opt.name:
        name = opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False
            
            
    
    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    
    # model
    ckpt_name = opt.ckpt_name
    ckpt_path = logdir / 'checkpoints' / f'{ckpt_name}.ckpt'
    config.model['params']['ckpt_path'] = ckpt_path
    model = instantiate_from_config(config.model)

    # data
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data Preparation Finished #####")
    
    return model, data, opt, logdir
