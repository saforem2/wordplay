# -*- coding: utf-8 -*-
"""
llm/__main__.py

Contains main entry point for training.
"""
from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals
)
import os
# import sys

import hydra
from pathlib import Path
import json
import logging
# from omegaconf import OmegaConf
from dataclasses import asdict
# from enrich import get_logger
# from ezpz import get_logger

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup, setup_wandb

from wordplay.configs import ExperimentConfig, PROJECT_ROOT
from wordplay.trainer import Trainer
try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None

# log = get_logger(__name__, "DEBUG")
log = logging.getLogger(__name__)


def include_file(f) -> bool:
    fp = Path(f)
    exclude_ = (
        'venv/' not in fp.as_posix()
        and 'old/' not in fp.as_posix()
        and 'outputs/' not in fp.as_posix()
        and 'wandb/' not in fp.as_posix()
        and 'data/' not in fp.as_posix()
        and 'cache/' not in fp.as_posix()
        and fp.suffix not in ['.pt', '.pth']
    )
    include_ = fp.suffix in ['.py', '.log', '.yaml']
    # return (
    #     exclude_ and include_
    #     # 'venv' not in fp.as_posix()
    #     # and fp.suffix in ['.py', '.log', '.yaml']
    # )
    return (exclude_ and include_)


def build_trainer(cfg: DictConfig) -> Trainer:
    rank = setup(
        framework=cfg.train.framework,
        backend=cfg.train.backend,
        seed=cfg.train.seed,
    )
    config: ExperimentConfig = instantiate(cfg)
    if rank != 0:
        log.setLevel("ERROR")
    else:
        log.setLevel("DEBUG")
        if config.train.use_wandb:
            setup_wandb(
                project_name=config.train.wandb_project,
                config=cfg,
            )
            if wandb is not None and wandb.run is not None:
                wandb.run.config['tokens_per_iter'] = config.tokens_per_iter
                wandb.run.config['samples_per_iter'] = config.samples_per_iter
        log.warning(json.dumps(asdict(config), indent=4))
    log.warning(f'Output dir: {os.getcwd()}')
    return Trainer(config)


def train(cfg: DictConfig) -> Trainer:
    trainer = build_trainer(cfg)
    trainer.train()
    if wandb is not None and wandb.run is not None:
        wandb.run.log_code(PROJECT_ROOT, include_fn=include_file)
        trainer.save_ckpt(add_to_wandb=True)
    return trainer


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> Trainer:
    return train(cfg)


if __name__ == '__main__':
    import wandb
    rank = main()
    if wandb.run is not None:
        wandb.finish(0)
    # sys.exit(0)
