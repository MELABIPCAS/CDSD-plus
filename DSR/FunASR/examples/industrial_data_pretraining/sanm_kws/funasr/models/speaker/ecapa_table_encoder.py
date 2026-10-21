#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAPATableEncoder(nn.Module):
    """
    简单的说话人嵌入查表模块：
    - 利用 scripts/cdsd/build_spk2emb.py 生成的 spk2emb.pt 初始化
    - 支持阶段 1 冻结、阶段 2 解冻继续微调
    """

    def __init__(
        self,
        table_path: str,
        trainable: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        data: Dict = torch.load(table_path, map_location="cpu")
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(f"{table_path} 中找不到 embeddings")
        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=not trainable, padding_idx=None
        )
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_trainable(self, trainable: bool):
        self.embedding.weight.requires_grad_(trainable)

    def forward(self, speaker_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(speaker_ids)
        emb = self.dropout(emb)
        if self.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

