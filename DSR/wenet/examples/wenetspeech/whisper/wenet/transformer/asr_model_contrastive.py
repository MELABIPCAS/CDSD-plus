# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
# Licensed under the Apache License, Version 2.0

from typing import Dict, List, Optional, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import IGNORE_ID


class ASRModelWithContrastive(ASRModel):
    """CTC-attention hybrid Encoder-Decoder model with contrastive learning"""

    def __init__(
        self,
        vocab_size: int,
        encoder: BaseEncoder,  # 这里可以是ConformerEncoder或TransformerEncoder
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
        apply_non_blank_embedding: bool = False,
        contrastive_learning: bool = False,
        contrastive_conf: Optional[dict] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            reverse_weight=reverse_weight,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            special_tokens=special_tokens,
            apply_non_blank_embedding=apply_non_blank_embedding,
        )
        
        self.contrastive_learning = contrastive_learning
        if contrastive_learning and contrastive_conf is not None:
            self.contrastive_weight = contrastive_conf.get('weight', 0.1)
            self.temperature = contrastive_conf.get('temperature', 0.1)
            self.projection_dim = contrastive_conf.get('projection_dim', 256)
            self.sample_strategy = contrastive_conf.get('sample_strategy', 'same_speaker')
            
            # 添加对比学习投影层
            self.contrastive_proj = torch.nn.Linear(
                encoder.output_size, self.projection_dim
            )
            # 添加对比学习分类头
            self.contrastive_head = torch.nn.Linear(
                self.projection_dim, 2  # 二分类：正常 vs 构音障碍
            )
        else:
            self.contrastive_weight = 0.0

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss + Contrastive learning"""
        # 调用父类的forward方法
        result = super().forward(batch, device)
        
        # 添加对比学习
        if self.contrastive_learning and 'type' in batch:
            loss_contrastive = self._calc_contrastive_loss(
                batch, device
            )
            if loss_contrastive is not None:
                result["loss_contrastive"] = loss_contrastive
                result["loss"] = result["loss"] + self.contrastive_weight * loss_contrastive
        
        return result

    def _calc_contrastive_loss(
        self,
        batch: dict,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """计算对比学习损失"""
        try:
            # 获取encoder输出（需要重新计算，因为父类方法中已经使用过了）
            speech = batch['feats'].to(device)
            speech_lengths = batch['feats_lengths'].to(device)
            encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
            
            batch_size, seq_len, hidden_dim = encoder_out.shape
            
            # 投影到对比学习空间
            proj_out = self.contrastive_proj(encoder_out)  # [B, T, projection_dim]
            
            # 获取样本类型标签
            sample_types = batch.get('type', [])
            if isinstance(sample_types, list) and len(sample_types) == batch_size:
                # 将类型转换为tensor
                type_labels = torch.tensor([
                    0 if t == 'normal' else 1 for t in sample_types
                ], device=device)
            else:
                # 如果没有类型信息，返回None
                return None
            
            # 全局平均池化
            mask = encoder_mask.squeeze(1).float()  # [B, T]
            pooled_out = (proj_out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            
            # 分类损失
            logits = self.contrastive_head(pooled_out)  # [B, 2]
            loss_contrastive = torch.nn.functional.cross_entropy(logits, type_labels)
            
            return loss_contrastive
        except Exception as e:
            print(f"对比学习计算失败: {e}")
            return None

    def _sample_contrastive_pairs(
        self,
        proj_out: torch.Tensor,
        batch: dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样对比学习的正负样本对"""
        batch_size = proj_out.shape[0]
        sample_types = batch.get('type', [])
        speakers = batch.get('speaker', [])
        
        if self.sample_strategy == "same_speaker":
            # 同说话人的正常vs异常对比
            pos_pairs = []
            neg_pairs = []
            labels = []
            
            for i in range(batch_size):
                if i < len(sample_types) and i < len(speakers):
                    current_type = sample_types[i]
                    current_speaker = speakers[i]
                    
                    # 寻找同说话人的不同类型样本
                    for j in range(batch_size):
                        if (j != i and j < len(sample_types) and j < len(speakers) and
                            speakers[j] == current_speaker and sample_types[j] != current_type):
                            pos_pairs.append(proj_out[i])
                            neg_pairs.append(proj_out[j])
                            labels.append(1 if current_type == 'normal' else 0)
                            break
            
            if len(pos_pairs) > 0:
                pos_pairs = torch.stack(pos_pairs)
                neg_pairs = torch.stack(neg_pairs)
                labels = torch.tensor(labels, device=device)
            else:
                # 如果没有找到配对，返回空tensor
                pos_pairs = torch.empty(0, proj_out.shape[-1], device=device)
                neg_pairs = torch.empty(0, proj_out.shape[-1], device=device)
                labels = torch.empty(0, dtype=torch.long, device=device)
        
        elif self.sample_strategy == "same_content":
            # 同内容的正常vs异常对比（需要文本内容相同）
            # 这里需要根据实际数据格式实现
            pos_pairs = torch.empty(0, proj_out.shape[-1], device=device)
            neg_pairs = torch.empty(0, proj_out.shape[-1], device=device)
            labels = torch.empty(0, dtype=torch.long, device=device)
        
        else:
            # 默认策略：batch内随机采样
            pos_pairs = proj_out
            neg_pairs = proj_out[torch.randperm(batch_size)]
            labels = torch.ones(batch_size, device=device)
        
        return pos_pairs, neg_pairs, labels

    def _contrastive_loss_fn(
        self,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算对比学习损失函数"""
        if pos_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=pos_pairs.device)
        
        # 计算相似度
        pos_sim = torch.cosine_similarity(pos_pairs, pos_pairs, dim=-1)
        neg_sim = torch.cosine_similarity(pos_pairs, neg_pairs, dim=-1)
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        logits = logits / self.temperature
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss