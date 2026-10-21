#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持 Cross-Attention 聚合多个短音频片段的说话人编码器
根据文档建议：使用多个短片段聚合说话人嵌入，而不是单条音频
"""

from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.register import tables
from funasr.models.transformer.utils.nets_utils import make_pad_mask


class CrossAttentionSpeakerEncoder(nn.Module):
    """
    使用 Cross-Attention 聚合多个短音频片段的说话人特征
    支持：
    1. 加权机制（长音频权重更高）
    2. 不确定性估计（对极短音频降低权重）
    3. 多片段聚合
    """
    
    def __init__(
        self,
        table_path: str,
        trainable: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        embed_dim: int = 256,
        num_heads: int = 4,
        use_length_weight: bool = True,  # 是否使用长度加权
        use_uncertainty: bool = True,  # 是否使用不确定性估计
        min_length_threshold: int = 50,  # 最小长度阈值（帧数），低于此值会降低权重
    ):
        super().__init__()
        
        # 加载预训练的说话人嵌入表
        data: Dict = torch.load(table_path, map_location="cpu")
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"{table_path} 中找不到 embeddings")
        
        self.base_embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=not trainable, padding_idx=None
        )
        self.embed_dim = embed_dim if embed_dim > 0 else embeddings.shape[-1]
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Cross-Attention 模块：用于聚合多个短音频片段
        self.num_heads = num_heads
        self.use_length_weight = use_length_weight
        self.use_uncertainty = use_uncertainty
        self.min_length_threshold = min_length_threshold
        
        # Query: 从基础嵌入表查询
        # Key/Value: 从多个短音频片段聚合
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, embed_dim)
        )
        
        # 特征投影层（用于将encoder输出的特征维度投影到embed_dim）
        # 延迟初始化：在第一次forward时根据实际特征维度创建
        self.feature_proj = None
        self._feature_dim = None
        
        # 不确定性估计网络（用于降低极短音频的权重）
        if use_uncertainty:
            self.uncertainty_net = nn.Sequential(
                nn.Linear(1, 32),  # 输入：音频长度（归一化）
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),  # 输出：不确定性权重 [0, 1]
            )
        
        # 长度加权（长音频权重更高）
        if use_length_weight:
            self.length_weight_net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),  # 输出：长度权重（>0）
            )
    
    def set_trainable(self, trainable: bool):
        """设置嵌入表是否可训练"""
        self.base_embedding.weight.requires_grad_(trainable)
    
    def compute_uncertainty_weight(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        计算不确定性权重：极短音频权重降低
        Args:
            lengths: (batch,) 音频长度（帧数）
        Returns:
            weights: (batch,) 不确定性权重 [0, 1]
        """
        # 确保长度至少为1，避免全0
        lengths = torch.clamp(lengths.float(), min=1.0)
        
        # 归一化长度到 [0, 1]，使用min_length_threshold作为参考
        # 如果所有长度都很小，使用min_length_threshold作为分母
        max_len = max(lengths.max().item(), self.min_length_threshold, 1.0)
        normalized_lengths = (lengths / max_len).clamp(0, 1).unsqueeze(-1)  # (batch, 1)
        
        # 通过网络计算不确定性权重（短音频 -> 低权重）
        uncertainty = self.uncertainty_net(normalized_lengths).squeeze(-1)  # (batch,)
        return uncertainty
    
    def compute_length_weight(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        计算长度权重：长音频权重更高
        Args:
            lengths: (batch,) 音频长度（帧数）
        Returns:
            weights: (batch,) 长度权重（>0）
        """
        # 确保长度至少为1，避免全0
        lengths = torch.clamp(lengths.float(), min=1.0)
        
        # 归一化长度，使用实际最大长度或至少为1
        max_len = max(lengths.max().item(), 1.0)
        normalized_lengths = (lengths / max_len).clamp(0, 1).unsqueeze(-1)  # (batch, 1)
        
        # 通过网络计算长度权重
        weights = self.length_weight_net(normalized_lengths).squeeze(-1)  # (batch,)
        # 确保权重至少为一个小正值，避免完全为0
        weights = torch.clamp(weights, min=0.1)
        return weights
    
    def forward(
        self,
        speaker_ids: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：支持单条音频或聚合多条短音频
        
        Args:
            speaker_ids: (batch,) 说话人ID
            audio_features: (batch, time, dim) 可选，多个短音频的特征（用于Cross-Attention聚合）
            audio_lengths: (batch,) 可选，音频长度（用于加权和不确定性估计）
        
        Returns:
            speaker_emb: (batch, embed_dim) 聚合后的说话人嵌入
        """
        # 从嵌入表获取基础说话人嵌入
        base_emb = self.base_embedding(speaker_ids)  # (batch, embed_dim)
        base_emb = self.dropout(base_emb)
        
        # 如果没有提供音频特征，直接返回基础嵌入
        if audio_features is None or audio_lengths is None:
            if self.normalize:
                base_emb = F.normalize(base_emb, dim=-1)
            return base_emb
        
        # 使用 Cross-Attention 聚合多个短音频片段
        # Query: 基础嵌入 (1, batch, embed_dim)
        query = base_emb.unsqueeze(0)  # (1, batch, embed_dim)
        
        # Key/Value: 音频特征 (time, batch, embed_dim)
        # 需要将 audio_features 从 (batch, time, dim) 转为 (time, batch, dim)
        if audio_features.dim() == 3:
            # 如果维度不匹配，需要投影
            if audio_features.size(-1) != self.embed_dim:
                # 延迟初始化投影层
                if self.feature_proj is None or self._feature_dim != audio_features.size(-1):
                    self.feature_proj = nn.Linear(audio_features.size(-1), self.embed_dim).to(audio_features.device)
                    self._feature_dim = audio_features.size(-1)
                audio_features = self.feature_proj(audio_features)
            kv = audio_features.transpose(0, 1)  # (time, batch, dim)
        else:
            kv = audio_features
        
        # 创建 mask（过滤 padding）
        if audio_lengths is not None:
            max_len = kv.size(0)
            mask = make_pad_mask(audio_lengths, maxlen=max_len)  # (batch, time)
            attn_mask = mask.to(kv.device)  # (batch, time)
        else:
            attn_mask = None
        
        # Cross-Attention
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=attn_mask,  # True 表示需要 mask
        )  # attn_out: (1, batch, embed_dim)
        
        aggregated_emb = attn_out.squeeze(0)  # (batch, embed_dim)
        
        # 应用加权机制
        if self.use_length_weight and audio_lengths is not None:
            # 确保长度有效
            valid_lengths = torch.clamp(audio_lengths, min=1)
            length_weights = self.compute_length_weight(valid_lengths)  # (batch,)
            aggregated_emb = aggregated_emb * length_weights.unsqueeze(-1)
        
        # 应用不确定性估计（降低极短音频的权重）
        if self.use_uncertainty and audio_lengths is not None:
            # 确保长度有效
            valid_lengths = torch.clamp(audio_lengths, min=1)
            uncertainty_weights = self.compute_uncertainty_weight(valid_lengths)  # (batch,)
            aggregated_emb = aggregated_emb * uncertainty_weights.unsqueeze(-1)
        
        # 融合基础嵌入和聚合嵌入
        # 如果音频特征有效（长度>阈值），使用聚合结果；否则回退到基础嵌入
        if audio_lengths is not None:
            # 确保长度至少为1，避免全0导致的mask问题
            valid_lengths = torch.clamp(audio_lengths, min=1)
            valid_mask = (valid_lengths >= self.min_length_threshold).float().unsqueeze(-1)  # (batch, 1)
            # 对于长度不足的样本，直接使用基础嵌入（不进行聚合）
            final_emb = valid_mask * aggregated_emb + (1 - valid_mask) * base_emb
        else:
            final_emb = aggregated_emb
        
        # 归一化
        if self.normalize:
            final_emb = F.normalize(final_emb, dim=-1)
        
        return final_emb


# 注册到 tables
@tables.register("speaker_encoder_classes", "CrossAttentionSpeakerEncoder")
class CrossAttentionSpeakerEncoderRegistered(CrossAttentionSpeakerEncoder):
    """注册版本，保持接口兼容"""
    pass

