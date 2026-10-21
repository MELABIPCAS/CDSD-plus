#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reptile Meta-Learning Trainer for Speaker-Independent ASR
每个说话人作为一个任务，通过元学习学习说话人无关特征
"""

import logging
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from funasr.train_utils.trainer import Trainer


class ReptileTrainer(Trainer):
    """
    Reptile元学习训练器
    将每个说话人作为一个任务，通过元学习学习说话人无关的ASR特征
    """
    
    def __init__(
        self,
        inner_lr: float = 1e-2,  # 内循环学习率
        inner_steps: int = 1,    # 内循环步数
        reptile_alpha: float = 1.0,  # Reptile更新权重
        speakers_per_batch: int = 4,  # 每个batch包含的说话人数量
        samples_per_speaker: int = 4,  # 每个说话人采样的样本数
        **kwargs
    ):
        """
        Args:
            inner_lr: 内循环（inner loop）学习率
            inner_steps: 每个任务的内循环更新步数
            reptile_alpha: Reptile算法的更新权重
            speakers_per_batch: 每个meta-batch包含的说话人数量
            samples_per_speaker: 每个说话人采样的样本数
        """
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.reptile_alpha = reptile_alpha
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        
        logging.info(f"ReptileTrainer initialized: inner_lr={inner_lr}, inner_steps={inner_steps}, "
                    f"reptile_alpha={reptile_alpha}, speakers_per_batch={speakers_per_batch}, "
                    f"samples_per_speaker={samples_per_speaker}")
    
    def _group_by_speaker(self, batch: Dict) -> Dict[int, List[Dict]]:
        """
        按说话人ID分组batch数据
        
        Args:
            batch: 包含'speaker_ids'字段的batch
            
        Returns:
            speaker_groups: {speaker_id: [samples]}
        """
        speaker_groups = defaultdict(list)
        
        # 获取batch size
        if isinstance(batch.get('speech'), torch.Tensor):
            batch_size = batch['speech'].shape[0]
        elif isinstance(batch.get('speech_lengths'), torch.Tensor):
            batch_size = batch['speech_lengths'].shape[0]
        else:
            # 尝试从其他字段推断
            for key in ['text', 'text_lengths', 'key']:
                if key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch_size = batch[key].shape[0] if batch[key].dim() > 0 else 1
                        break
                    elif isinstance(batch[key], list):
                        batch_size = len(batch[key])
                        break
            else:
                return speaker_groups
        
        # 获取speaker_ids
        speaker_ids = batch.get('speaker_ids', None)
        if speaker_ids is None:
            # 如果没有speaker_ids，尝试从key中提取
            keys = batch.get('key', [])
            if keys and isinstance(keys, list):
                speaker_ids = []
                for key in keys:
                    # 尝试提取说话人ID（假设前4个字符是说话人ID）
                    try:
                        if isinstance(key, str):
                            spk_id = int(key[:4])
                        else:
                            spk_id = 0
                        speaker_ids.append(spk_id)
                    except:
                        speaker_ids.append(0)  # 默认值
                speaker_ids = torch.tensor(speaker_ids)
            else:
                # 如果都没有，返回空分组
                return speaker_groups
        
        # 确保speaker_ids是tensor
        if not isinstance(speaker_ids, torch.Tensor):
            speaker_ids = torch.tensor(speaker_ids)
        
        # 确保speaker_ids长度匹配batch_size
        if speaker_ids.shape[0] != batch_size:
            logging.warning(f"speaker_ids length {speaker_ids.shape[0]} != batch_size {batch_size}")
            return speaker_groups
        
        # 按说话人分组
        for i in range(batch_size):
            spk_id = int(speaker_ids[i].item())
            sample = {}
            for key, value in batch.items():
                if key == 'speaker_ids':
                    continue  # 跳过speaker_ids本身
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] > i:
                        sample[key] = value[i]
                    elif value.dim() == 0:
                        sample[key] = value
                elif isinstance(value, list):
                    if i < len(value):
                        sample[key] = value[i]
                else:
                    sample[key] = value
            speaker_groups[spk_id].append(sample)
        
        return speaker_groups
    
    def _sample_speaker_batch(
        self, 
        speaker_groups: Dict[int, List[Dict]], 
        speakers_per_batch: int,
        samples_per_speaker: int
    ) -> List[Tuple[int, List[Dict]]]:
        """
        采样说话人和对应的样本
        
        Args:
            speaker_groups: 按说话人分组的数据
            speakers_per_batch: 每个batch的说话人数量
            samples_per_speaker: 每个说话人的样本数
            
        Returns:
            selected_speakers: [(speaker_id, [samples]), ...]
        """
        available_speakers = [spk_id for spk_id, samples in speaker_groups.items() 
                            if len(samples) >= samples_per_speaker]
        
        if len(available_speakers) < speakers_per_batch:
            # 如果可用说话人不足，使用所有可用说话人
            selected_speakers = available_speakers
        else:
            # 随机选择说话人
            selected_speakers = np.random.choice(
                available_speakers, 
                size=min(speakers_per_batch, len(available_speakers)), 
                replace=False
            ).tolist()
        
        result = []
        for spk_id in selected_speakers:
            samples = speaker_groups[spk_id]
            # 随机采样
            if len(samples) > samples_per_speaker:
                indices = np.random.choice(len(samples), samples_per_speaker, replace=False)
                selected_samples = [samples[i] for i in indices]
            else:
                selected_samples = samples
            result.append((spk_id, selected_samples))
        
        return result
    
    def _collate_speaker_samples(self, samples: List[Dict]) -> Dict:
        """
        将同一说话人的样本collate成batch
        
        Args:
            samples: 同一说话人的样本列表
            
        Returns:
            batched_data: collated batch
        """
        if not samples:
            return {}
        
        batched = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            if isinstance(values[0], torch.Tensor):
                try:
                    batched[key] = torch.stack(values)
                except:
                    # 如果无法stack，尝试pad_sequence
                    batched[key] = torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True
                    )
            elif isinstance(values[0], (int, float)):
                batched[key] = torch.tensor(values)
            else:
                batched[key] = values
        
        return batched
    
    def _inner_loop_update(
        self,
        model: nn.Module,
        speaker_samples: List[Dict],
        optim_inner: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        对单个说话人任务进行内循环更新
        
        Args:
            model: 模型
            speaker_samples: 该说话人的样本
            optim_inner: 内循环优化器（如果为None，使用手动梯度更新）
            
        Returns:
            (loss_dict, updated_params): 损失字典和更新后的参数
        """
        # 创建内循环优化器（如果未提供）
        if optim_inner is None:
            # 使用SGD作为内循环优化器
            optim_inner = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.inner_lr
            )
        
        # 保存初始参数
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 内循环更新
        for step in range(self.inner_steps):
            # 采样support set（用于内循环更新）
            if len(speaker_samples) > self.samples_per_speaker:
                support_samples = np.random.choice(
                    speaker_samples, 
                    size=self.samples_per_speaker, 
                    replace=False
                ).tolist()
            else:
                support_samples = speaker_samples
            
            # Collate support set
            support_batch = self._collate_speaker_samples(support_samples)
            
            # Forward pass
            model.train()
            loss_dict = model(**support_batch)
            
            # 计算总损失
            if isinstance(loss_dict, dict):
                total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            else:
                total_loss = loss_dict
            
            # Backward and update
            optim_inner.zero_grad()
            if hasattr(self, 'scaler') and self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optim_inner)
                self.scaler.update()
            else:
                total_loss.backward()
                optim_inner.step()
        
        # 获取更新后的参数
        updated_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 恢复初始参数（用于Reptile更新）
        for name, param in model.named_parameters():
            param.data.copy_(initial_params[name])
        
        return loss_dict, updated_params
    
    def train_epoch(
        self,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        **kwargs
    ):
        """
        Reptile元学习训练一个epoch
        
        Args:
            model: 模型
            optim: 外循环优化器
            scheduler: 学习率调度器
            scaler: 混合精度scaler
            epoch: 当前epoch
        """
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.scaler = scaler
        
        model.train()
        total_loss = 0.0
        total_steps = 0
        
        dataloader = kwargs.get('dataloader_train', None)
        if dataloader is None:
            raise ValueError("dataloader_train is required")
        
        # 收集所有数据并按说话人分组
        logging.info("Collecting data and grouping by speaker...")
        all_speaker_groups = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            speaker_groups = self._group_by_speaker(batch)
            for spk_id, samples in speaker_groups.items():
                all_speaker_groups[spk_id].extend(samples)
        
        logging.info(f"Found {len(all_speaker_groups)} speakers with data")
        
        # Reptile训练循环
        num_meta_batches = len(all_speaker_groups) // self.speakers_per_batch
        if num_meta_batches == 0:
            num_meta_batches = 1
        
        for meta_batch_idx in range(num_meta_batches):
            # 采样说话人
            selected_speakers = self._sample_speaker_batch(
                all_speaker_groups,
                self.speakers_per_batch,
                self.samples_per_speaker
            )
            
            if not selected_speakers:
                continue
            
            # 保存初始参数
            initial_params = {name: param.clone() for name, param in model.named_parameters()}
            
            # 对每个说话人任务进行内循环更新
            updated_params_list = []
            meta_loss = 0.0
            
            for spk_id, speaker_samples in selected_speakers:
                if len(speaker_samples) < self.samples_per_speaker:
                    continue
                
                # 内循环更新
                loss_dict, updated_params = self._inner_loop_update(
                    model, speaker_samples
                )
                
                # 累积损失
                if isinstance(loss_dict, dict):
                    meta_loss += sum(v.item() if isinstance(v, torch.Tensor) else v 
                                   for v in loss_dict.values())
                else:
                    meta_loss += loss_dict.item() if hasattr(loss_dict, 'item') else loss_dict
                
                updated_params_list.append(updated_params)
            
            if not updated_params_list:
                continue
            
            # Reptile更新：计算参数更新方向
            param_updates = {}
            for name in initial_params.keys():
                # 计算所有任务的平均更新
                avg_update = torch.zeros_like(initial_params[name])
                for updated_params in updated_params_list:
                    avg_update += (updated_params[name] - initial_params[name])
                avg_update /= len(updated_params_list)
                
                # Reptile更新：θ = θ + α * (avg_update)
                param_updates[name] = self.reptile_alpha * avg_update
            
            # 应用Reptile更新
            for name, param in model.named_parameters():
                if name in param_updates:
                    param.data.add_(param_updates[name])
            
            # 外循环更新（可选，用于进一步优化）
            # 使用query set计算损失并更新
            query_samples = []
            for spk_id, speaker_samples in selected_speakers:
                if len(speaker_samples) >= self.samples_per_speaker * 2:
                    # 使用剩余的样本作为query set
                    query_indices = np.random.choice(
                        len(speaker_samples),
                        size=self.samples_per_speaker,
                        replace=False
                    )
                    query_samples.extend([speaker_samples[i] for i in query_indices])
            
            if query_samples:
                query_batch = self._collate_speaker_samples(query_samples)
                model.train()
                query_loss_dict = model(**query_batch)
                
                if isinstance(query_loss_dict, dict):
                    query_loss = sum(v for v in query_loss_dict.values() 
                                   if isinstance(v, torch.Tensor))
                else:
                    query_loss = query_loss_dict
                
                # 外循环backward
                optim.zero_grad()
                if scaler is not None:
                    scaler.scale(query_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    query_loss.backward()
                    optim.step()
                
                total_loss += query_loss.item()
            
            total_steps += 1
            
            # 日志记录
            if (meta_batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(total_steps, 1)
                logging.info(
                    f"Epoch {epoch}, Meta-batch {meta_batch_idx + 1}/{num_meta_batches}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
        
        return total_loss / max(total_steps, 1)
    
    def val_epoch(self, model=None, dataloader_val=None, epoch=0, **kwargs):
        """
        验证epoch（使用标准验证方法）
        """
        # 使用父类的验证方法
        return super().val_epoch(
            model=model,
            dataloader_val=dataloader_val,
            epoch=epoch,
            **kwargs
        )
    
    def save_checkpoint(self, model=None, optim=None, scheduler=None, epoch=0, loss=None, **kwargs):
        """
        保存checkpoint（使用标准保存方法）
        """
        # 使用父类的保存方法
        return super().save_checkpoint(
            model=model,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
            **kwargs
        )

