#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reptile Meta-Learning Training Script for Speaker-Independent ASR
基于train.py，但使用ReptileTrainer进行元学习训练
"""

import os
import sys
import logging
import time
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import ShardedGradScaler

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from funasr.register import tables
from funasr.models.auto.auto_model import AutoModel
from funasr.train_utils.trainer import Trainer

# 确保导入ReptileTrainer和AudioDatasetSpeaker
try:
    from funasr.train_utils.reptile_trainer import ReptileTrainer
except ImportError:
    logging.warning("ReptileTrainer not found, will use standard Trainer")
    ReptileTrainer = None

try:
    from funasr.datasets.audio_datasets.datasets_spk import AudioDatasetSpeaker
except ImportError:
    logging.warning("AudioDatasetSpeaker not found")
    AudioDatasetSpeaker = None

from funasr.utils.misc import set_all_random_seed
from funasr.utils.model_utils import prepare_model_dir

try:
    from torch.utils.tensorboard import SummaryWriter
    writer = None
except:
    writer = None

import hydra
from omegaconf import DictConfig, OmegaConf


def main_hydra():
    @hydra.main(config_path=None, config_name=None, version_base=None)
    def decorated_main(cfg: DictConfig) -> None:
        logging.info("Reptile Meta-Learning Training")
        logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        
        kwargs = OmegaConf.to_container(cfg, resolve=True)
        main(**kwargs)
    
    decorated_main()


def main(**kwargs):
    """主训练函数"""
    # 设置随机种子
    set_all_random_seed(kwargs.get("seed", 0))
    torch.backends.cudnn.enabled = kwargs.get("cudnn_enabled", torch.backends.cudnn.enabled)
    torch.backends.cudnn.benchmark = kwargs.get("cudnn_benchmark", torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = kwargs.get("cudnn_deterministic", True)
    torch.backends.cuda.matmul.allow_tf32 = kwargs.get("enable_tf32", True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        tables.print()
    
    # 检查是否使用DDP或FSDP
    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    use_fsdp = kwargs.get("use_fsdp", False)
    
    if use_ddp or use_fsdp:
        dist.init_process_group(backend=kwargs.get("backend", "nccl"), init_method="env://")
        torch.cuda.set_device(local_rank)

    logging.info("Build model, frontend, tokenizer")
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "cpu"
    model = AutoModel(**kwargs)

    # 保存config.yaml
    if (
        (use_ddp or use_fsdp)
        and dist.get_rank() == 0
        or not (use_ddp or use_fsdp)
        and local_rank == 0
    ):
        prepare_model_dir(**kwargs)

    # 解析kwargs
    kwargs = model.kwargs
    kwargs["device"] = device
    tokenizer = kwargs["tokenizer"]

    # 构建优化器和调度器
    logging.info("Build optim")
    optim = model.build_optim(**kwargs)
    logging.info("Build scheduler")
    scheduler = model.build_scheduler(**kwargs)

    # 构建数据加载器
    logging.info("Build dataloader")
    dataloader_class = tables.dataloader_classes.get(
        kwargs.get("dataloader", "DataloaderMapStyle")
    )
    dataloader = dataloader_class(**kwargs)
    dataloader_tr, dataloader_val = dataloader.build_iter(epoch=0)

    # 构建ReptileTrainer
    train_conf = kwargs.get("train_conf", {})
    trainer_class_name = train_conf.get("trainer_class", "ReptileTrainer")
    
    if trainer_class_name == "ReptileTrainer" and ReptileTrainer is not None:
        # 提取Reptile特定参数
        reptile_params = {
            'inner_lr': train_conf.pop('reptile_inner_lr', 0.01),
            'inner_steps': train_conf.pop('reptile_inner_steps', 1),
            'reptile_alpha': train_conf.pop('reptile_alpha', 1.0),
            'speakers_per_batch': train_conf.pop('reptile_speakers_per_batch', 4),
            'samples_per_speaker': train_conf.pop('reptile_samples_per_speaker', 4),
        }
        trainer = ReptileTrainer(
            local_rank=local_rank,
            use_ddp=use_ddp,
            use_fsdp=use_fsdp,
            device=device,
            output_dir=kwargs.get("output_dir", "./exp"),
            **reptile_params,
            **train_conf,
        )
        logging.info(f"Using ReptileTrainer with params: {reptile_params}")
    else:
        # 回退到标准Trainer
        if ReptileTrainer is None:
            logging.warning("ReptileTrainer not available, using standard Trainer")
        trainer = Trainer(
            local_rank=local_rank,
            use_ddp=use_ddp,
            use_fsdp=use_fsdp,
            device=device,
            output_dir=kwargs.get("output_dir", "./exp"),
            **train_conf,
        )

    scaler = GradScaler(enabled=trainer.use_fp16) if trainer.use_fp16 else None
    scaler = ShardedGradScaler(enabled=trainer.use_fp16) if trainer.use_fsdp else scaler

    # TensorBoard
    tensorboard_dir = os.path.join(kwargs.get("output_dir", "./exp"), "tensorboard")
    try:
        if (use_ddp or use_fsdp) and dist.get_rank() == 0 or not (use_ddp or use_fsdp):
            writer = SummaryWriter(tensorboard_dir)
        else:
            writer = None
    except:
        writer = None

    # 训练循环
    dataloader_tr, dataloader_val = None, None
    for epoch in range(trainer.start_epoch, trainer.max_epoch):
        time1 = time.perf_counter()

        # 重新构建数据加载器（如果需要）
        if dataloader_tr is None or epoch % train_conf.get("reload_dataloader_epoch", 1) == 0:
            dataloader_tr, dataloader_val = dataloader.build_iter(epoch=epoch)

        # 训练一个epoch
        logging.info(f"Epoch {epoch}/{trainer.max_epoch}")
        
        if isinstance(trainer, ReptileTrainer) if ReptileTrainer else False:
            # 使用Reptile训练
            train_loss = trainer.train_epoch(
                model=model,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                dataloader_train=dataloader_tr,
            )
        else:
            # 使用标准训练
            train_loss = trainer.train_epoch(
                model=model,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                dataloader_train=dataloader_tr,
            )

        # 验证
        if dataloader_val is not None:
            val_loss = trainer.val_epoch(
                model=model,
                dataloader_val=dataloader_val,
                epoch=epoch,
            )
        else:
            val_loss = None

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 保存模型
        trainer.save_checkpoint(
            model=model,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            loss=train_loss,
        )

        # 记录日志
        time2 = time.perf_counter()
        logging.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f if val_loss else 'N/A'}, "
            f"time={time2-time1:.2f}s"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar("val/loss", val_loss, epoch)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main_hydra()

