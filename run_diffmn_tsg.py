# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import time
os.environ['HYDRA_FULL_ERROR'] = '6'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

import os, sys
import time

from pytorch_lightning.trainer import Trainer
from utils_diffmn.cli_utils import get_parser
from utils_diffmn.init_utils import init_model_data_trainer
from utils_diffmn.test_utils import test_model_with_diffcde


if __name__ == "__main__":
    sys.argv = [
        "--base", "configs/Diff_MN.yaml",
        "--gpus", "0,",
        "--logdir", "./logs/",
        "-sl", "28",
        "-up",
        "-nl", "16",
        "--batch_size", "256",
        "-lr", "0.0001",
        "-s", "0"
    ]


    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)
    train_loader = data.train_dataloader()
    train_loader_eval=data.train_dataloader_eval()
    opt.use_label=False
    opt.train = True

    opt.load_generation=False

    if opt.train:
        try:
            trainer.logger.experiment.config.update(opt)
            trainer.fit(model, data)
            test_model_with_diffcde(model, data, trainer, opt, logdir, train_loader_eval,tag='best')
        except Exception:
            melk()
            raise


