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
import hydra
import json
import logging

import ezpz as ez

from pathlib import Path
from dataclasses import asdict
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

# from ezpz.dist import setup, setup_wandb

from wordplay.configs import HERE, ExperimentConfig, PROJECT_ROOT
from wordplay.trainer import Trainer

RANK = ez.get_rank()
WORLD_SIZE = ez.get_world_size()

# log = get_logger(__name__, "DEBUG")
log = logging.getLogger(__name__)
log.setLevel("INFO") if RANK == 0 else log.setLevel("CRITICAL")


def include_file(f: str | Path) -> bool:
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
    return (exclude_ and include_)


def setup_training(cfg: DictConfig) -> ExperimentConfig:
    rank = ez.setup(
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
            os.environ["WANDB_CACHE_DIR"] = PROJECT_ROOT.joinpath(
                ".cache",
                "wandb"
            ).as_posix()
            try:
                import wandb
                HAS_WANDB = True
                _ = ez.setup_wandb(
                    project_name=config.train.wandb_project,
                    config=cfg,
                )
            except Exception as exc:
                wandb = None
                HAS_WANDB = False
                if isinstance(exc, wandb.errors.UnsupportedError):
                    log.critical(' '.join([
                        "Unsupported wandb version.",
                        "Update with 'python3 -m pip install --upgrade wandb'"
                    ]))
                log.warning(f'Continuing without wandb!!')
            if wandb is not None and wandb.run is not None and HAS_WANDB:
                wandb.run.config['tokens_per_iter'] = config.tokens_per_iter
                wandb.run.config['samples_per_iter'] = config.samples_per_iter
        log.warning(json.dumps(asdict(config), indent=4))
    log.warning(f'Output dir: {os.getcwd()}')
    return config


def build_trainer(config: ExperimentConfig) -> Trainer:
    return Trainer(config)


def generate_text(
        trainer: Trainer,
        query: str,
        num_samples: int = 1,
        max_new_tokens: int = 256,
        top_k: int = 16,
        display: bool = False
) -> dict:
    import ezpz
    if ezpz.get_rank() == 0:
        outputs = trainer.evaluate(
            query,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            display=display
        )
        log.info(f"['prompt']: '{query}'")
        log.info("['response']:\n\n" + fr"{outputs['0']['raw']}")
    # ---------------------------------------
    # NOTE: `outputs` is a dict of the form:
    # outputs = {
    #    "1": {
    #         "raw": str,
    #         "prompt": str,
    #         "formatted": str,
    #
    #     },
    #     ...,
    #     "num_samples-1": {
    #         "raw": str,
    #         "prompt": str,
    #         "formatted": str,
    #     }
    # }
    # ---------------------------------------
    return outputs


def train(cfg: DictConfig) -> Trainer:
    config: ExperimentConfig = setup_training(cfg)
    # trainer = build_trainer(config)
    trainer = Trainer(config)
    trainer.train()
    output = None
    if RANK == 0:
        output = generate_text(
            trainer=trainer,
            query="What is an LLM?",
            num_samples=1,
            max_new_tokens=256,
            top_k=16,
            display=False
        )
    if wandb is not None and HAS_WANDB and wandb.run is not None:
        trainer.save_ckpt(add_to_wandb=False)
    return trainer, output


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> Trainer:
    return train(cfg)


if __name__ == '__main__':
    import time
    t0 = time.perf_counter()
    os.environ['START_TIME'] = f'{t0}'
    rank = main()
