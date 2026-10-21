import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import re
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables


@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturb")
class SpeechPreprocessSpeedPerturb(nn.Module):
    def __init__(self, speed_perturb: list = None, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb

    def forward(self, waveform, fs, **kwargs):
        if self.speed_perturb is None:
            return waveform
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.view(1, -1), fs, [["speed", str(speed)], ["rate", str(fs)]]
            )
            waveform = waveform.view(-1)

        return waveform


@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturbStrongAug")
class SpeechPreprocessSpeedPerturbStrongAug(nn.Module):
    """
    在基础 SpeechPreprocessSpeedPerturb 的基础上：
    - 所有样本先做相同的 speed perturb
    - 对打了 strong_aug=True 标记的样本，再叠加一次“稍微更强”的时域掩码
      （为了简单起见，这里用 torchaudio 的简单 time mask 近似，幅度只比默认稍大一点点）
    """

    def __init__(self, speed_perturb: list = None, strong_time_mask_prob: float = 0.5, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb
        # strong_aug 样本叠加强 mask 的概率（避免每次都很重）
        self.strong_time_mask_prob = strong_time_mask_prob

    def forward(self, waveform, fs, strong_aug: bool = False, **kwargs):
        # 1) 常规 speed perturb，和原来一致
        if self.speed_perturb is not None:
            speed = random.choice(self.speed_perturb)
            if speed != 1.0:
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.tensor(waveform)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform.view(1, -1), fs, [["speed", str(speed)], ["rate", str(fs)]]
                )
                waveform = waveform.view(-1)

        # 2) 对 strong_aug=True 的样本，叠加一层“略微更强”的 time mask
        # 这里只对 wave 级时域做一个非常轻量遮挡（幅度只比 baseline 多一点点）
        if strong_aug and random.random() < self.strong_time_mask_prob:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            # 将 1D waveform 视作单通道特征，做一次简易时间遮挡
            length = waveform.shape[0]
            if length > 0:
                # 最大遮挡比例非常小，比如 2% 的长度
                max_width = max(1, int(length * 0.02))
                width = random.randint(1, max_width)
                start = random.randint(0, max(0, length - width))
                waveform = waveform.clone()
                waveform[start : start + width] = 0.0

        return waveform


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
    def __init__(
        self,
        seg_dict: str = None,
        text_cleaner: Collection[str] = None,
        split_with_space: bool = False,
        **kwargs
    ):
        super().__init__()

        self.text_cleaner = TextCleaner(text_cleaner)

    def forward(self, text, **kwargs):
        text = self.text_cleaner(text)

        return text


@tables.register("preprocessor_classes", "TextPreprocessDropPrefixScores")
class TextPreprocessDropPrefixScores(nn.Module):
    def __init__(self, expect_scores: int = 4, **kwargs):
        super().__init__()
        self.expect_scores = expect_scores

    def forward(self, text: str, **kwargs):
        if not isinstance(text, str):
            return text
        parts = text.strip().split()
        if len(parts) >= self.expect_scores + 1:
            def _is_float_token(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except Exception:
                    return False
            if all(_is_float_token(tok) for tok in parts[: self.expect_scores]):
                return " ".join(parts[self.expect_scores :])
        return text
