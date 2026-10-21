#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Dict, Optional

import torch

from funasr.register import tables
from funasr.datasets.audio_datasets.datasets import AudioDataset


def _load_spk_table(path: Optional[str]) -> Dict:
    if path is None:
        return {"speaker2idx": {}, "embeddings": None}
    data = torch.load(path, map_location="cpu")
    speaker2idx = data.get("speaker2idx")
    if speaker2idx is None:
        raise ValueError(f"spk2emb 文件 {path} 缺少 speaker2idx 字段")
    return data


@tables.register("dataset_classes", "AudioDatasetSpeaker")
class AudioDatasetSpeaker(AudioDataset):
    """
    基于 AudioDataset，额外输出 speaker_ids（int64）用于查表。
    说话人 ID 默认取 key 前四个字符，也可通过配置指定字段。
    """

    def __init__(
        self,
        path,
        speaker_table_path: Optional[str] = None,
        speaker_id_field: Optional[str] = None,
        speaker_id_prefix: int = 4,
        is_training: bool = True,
        **kwargs,
    ):
        self.speaker_id_field = speaker_id_field
        self.speaker_id_prefix = speaker_id_prefix
        self.is_training = is_training
        self.spk_table_info = _load_spk_table(speaker_table_path)
        self.speaker2idx = self.spk_table_info.get("speaker2idx", {}).copy()  # 使用copy避免修改原始字典
        super().__init__(path, **kwargs)

    def _resolve_speaker_idx(self, item: Dict) -> int:
        if self.speaker_id_field and self.speaker_id_field in item:
            speaker_id = item[self.speaker_id_field]
        else:
            key = item.get("key") or item.get("utt_id")
            if key is None:
                # 无法解析说话人，返回 -1（后续 loss 会忽略）
                return -1
            speaker_id = key[: self.speaker_id_prefix]
        
        if speaker_id not in self.speaker2idx:
            if self.is_training:
                # 训练时：若表中没有该说话人，则追加新索引（随机初始化）
                new_idx = len(self.speaker2idx)
                self.speaker2idx[speaker_id] = new_idx
            else:
                # 推理时：如果表中没有该说话人，返回 -1（表示未知说话人）
                # 这样模型会使用默认的说话人嵌入或忽略该样本
                return -1
        return self.speaker2idx[speaker_id]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self.index_ds[index]
        speaker_idx = self._resolve_speaker_idx(item)
        sample["speaker_ids"] = torch.tensor(speaker_idx, dtype=torch.int64)
        return sample

    def collator(self, samples=None):
        outputs = super().collator(samples)
        if "speaker_ids" in outputs:
            # 过滤空的 speaker_ids，避免 stack 失败
            valid = [t for t in outputs["speaker_ids"] if t.numel() > 0]
            if len(valid) == 0:
                valid = [torch.tensor([-1], dtype=torch.int64)]
            outputs["speaker_ids"] = torch.stack(valid, dim=0)
        return outputs

