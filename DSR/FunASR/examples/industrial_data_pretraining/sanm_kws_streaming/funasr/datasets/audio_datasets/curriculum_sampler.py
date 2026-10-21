"""
课程学习采样器（Curriculum Learning Sampler）
根据CER难度动态调整数据采样策略
"""
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Sampler, BatchSampler, SequentialSampler

from funasr.register import tables

logger = logging.getLogger(__name__)


def _load_index_info(index_file: Optional[str]) -> Dict:
    if not index_file:
        return {}
    path = Path(index_file)
    if not path.exists():
        raise FileNotFoundError(f"Curriculum index_file not found: {index_file}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_list(config_value, index_value):
    if config_value is not None:
        return config_value
    if index_value is not None:
        return index_value
    return []


@tables.register("batch_sampler_classes", "CurriculumSampler")
def CurriculumSampler_fn(dataset, **kwargs):
    """
    包装函数，返回 DataLoader 需要的 batch_sampler 设置。
    """
    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        raise ValueError("CurriculumSampler requires 'batch_size' in dataset_conf.")

    is_training = kwargs.get("is_training", True)

    if not is_training:
        # 验证集：支持单独的 valid_batch_size，用于控制显存占用
        valid_batch_size = kwargs.get("valid_batch_size", batch_size)
        seq_sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(
            seq_sampler, batch_size=valid_batch_size, drop_last=False
        )
        return {
            "batch_sampler": batch_sampler,
            "num_workers": kwargs.get("num_workers", 4),
            "pin_memory": kwargs.get("pin_memory", True),
        }

    index_info = _load_index_info(kwargs.get("index_file"))

    sampler = CurriculumSamplerCore(
        dataset=dataset,
        batch_size=batch_size,
        num_epochs=kwargs.get("num_epochs", 100),
        initial_synthetic_ratio=kwargs.get("initial_synthetic_ratio", 1.0),
        final_synthetic_ratio=kwargs.get("final_synthetic_ratio", 0.3),
        synthetic_ratio_start_epoch=kwargs.get("synthetic_ratio_start_epoch"),
        synthetic_ratio_end_epoch=kwargs.get("synthetic_ratio_end_epoch"),
        initial_difficulty_threshold=kwargs.get("initial_difficulty_threshold", 0.0),
        final_difficulty_threshold=kwargs.get("final_difficulty_threshold", 1.0),
        shuffle=kwargs.get("shuffle", True),
        balance_by_speaker=kwargs.get("balance_by_speaker", True),
        synthetic_indices=_resolve_list(
            kwargs.get("synthetic_indices"), index_info.get("synthetic_indices")
        ),
        real_indices=_resolve_list(
            kwargs.get("real_indices"), index_info.get("real_indices")
        ),
        synthetic_cer_scores=_resolve_list(
            kwargs.get("synthetic_cer_scores"), index_info.get("synthetic_cer_scores")
        ),
        real_cer_scores=_resolve_list(
            kwargs.get("real_cer_scores"), index_info.get("real_cer_scores")
        ),
        synthetic_speaker_indices=_resolve_list(
            kwargs.get("synthetic_speaker_indices"), index_info.get("synthetic_speaker_indices")
        ),
        real_speaker_indices=_resolve_list(
            kwargs.get("real_speaker_indices"), index_info.get("real_speaker_indices")
        ),
        drop_last=kwargs.get("drop_last", False),
        batch_type=kwargs.get("batch_type", "example"),
        # 额外参数：长度分桶大小，用于在保持整体易→难的前提下，
        # 在局部窗口内按长度友好打包，减少过多的小 batch 数量
        length_bucket_size=kwargs.get("length_bucket_size", 0),
        max_token_length=kwargs.get("max_token_length"),
        min_token_length=kwargs.get("min_token_length", 0),
        # 课程“加噪声”相关参数：用于从强课程逐步过渡到接近随机
        difficulty_noise_max=kwargs.get("difficulty_noise_max", 0.0),
        difficulty_noise_start_epoch=kwargs.get("difficulty_noise_start_epoch", 0),
        difficulty_noise_end_epoch=kwargs.get("difficulty_noise_end_epoch"),
        is_training=is_training,
    )

    dataloader_args = {
        "batch_sampler": sampler,
        "num_workers": kwargs.get("num_workers", 4),
        "pin_memory": kwargs.get("pin_memory", True),
    }
    return dataloader_args


class CurriculumSamplerCore(Sampler):
    """
    课程学习采样器

    特点：
    1. 根据CER从简单到困难排序
    2. 动态调整合成/真实数据比例
    3. 支持渐进式难度增加
    """

    def __init__(
        self,
        dataset,
        synthetic_indices: List[int],
        real_indices: List[int],
        synthetic_cer_scores: List[float],
        real_cer_scores: List[float],
        num_epochs: int = 100,
        initial_synthetic_ratio: float = 1.0,
        final_synthetic_ratio: float = 0.3,
        synthetic_ratio_start_epoch: int = 0,
        synthetic_ratio_end_epoch: Optional[int] = None,
        initial_difficulty_threshold: float = 0.0,
        final_difficulty_threshold: float = 1.0,
        shuffle: bool = True,
        balance_by_speaker: bool = True,
        synthetic_speaker_indices: Dict[str, List[int]] = None,
        real_speaker_indices: Dict[str, List[int]] = None,
        batch_size: int = 32,
        drop_last: bool = False,
        batch_type: str = "example",
        length_bucket_size: int = 0,
        max_token_length: Optional[int] = None,
        min_token_length: int = 0,
        is_training: bool = True,
        **kwargs,
    ):
        self.dataset = dataset
        self.synthetic_indices = synthetic_indices
        self.real_indices = real_indices
        self.synthetic_cer_scores = synthetic_cer_scores
        self.real_cer_scores = real_cer_scores
        self.num_epochs = max(1, num_epochs)
        self.initial_synthetic_ratio = initial_synthetic_ratio
        self.final_synthetic_ratio = final_synthetic_ratio
        # synthetic_ratio 的调度区间（类似 difficulty_noise_*）
        self.synthetic_ratio_start_epoch = int(synthetic_ratio_start_epoch or 0)
        if synthetic_ratio_end_epoch is None:
            synthetic_ratio_end_epoch = self.num_epochs
        self.synthetic_ratio_end_epoch = int(synthetic_ratio_end_epoch)
        self.initial_difficulty_threshold = initial_difficulty_threshold
        self.final_difficulty_threshold = final_difficulty_threshold
        self.shuffle = shuffle
        self.balance_by_speaker = balance_by_speaker
        self.synthetic_speaker_indices = synthetic_speaker_indices or {}
        self.real_speaker_indices = real_speaker_indices or {}
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.is_training = is_training
        self.batch_type = batch_type
        # 长度分桶大小：>0 时，在按 CER 排好序的前提下，
        # 先按固定窗口切分，再在窗口内按长度排序+打包，
        # 以减少由于长句扎堆导致的过多小 batch
        self.length_bucket_size = int(length_bucket_size) if length_bucket_size else 0
        self.max_token_length = max_token_length
        self.min_token_length = min_token_length
        self._last_batch_count = math.ceil(len(self.dataset) / max(1, self.batch_size))

        # 难度噪声相关参数：用于让“易→难”的顺序随着 epoch 逐步变松，
        # 最终接近随机顺序，便于从课程学习平滑过渡到无课程训练。
        self.difficulty_noise_max = float(kwargs.get("difficulty_noise_max", 0.0))
        self.difficulty_noise_start_epoch = int(
            kwargs.get("difficulty_noise_start_epoch", 0)
        )
        # 默认不填时：在整个 num_epochs 范围内逐渐加大噪声
        end_epoch = kwargs.get("difficulty_noise_end_epoch")
        if end_epoch is None:
            end_epoch = self.num_epochs
        self.difficulty_noise_end_epoch = int(end_epoch)

        if not self.synthetic_indices and not self.real_indices:
            total_len = len(self.dataset)
            self.synthetic_indices = list(range(total_len))
            self.real_indices = []
            self.synthetic_cer_scores = self.synthetic_cer_scores or [0.5] * total_len
            self.real_cer_scores = self.real_cer_scores or []

        assert len(self.synthetic_indices) == len(self.synthetic_cer_scores), (
            "Synthetic indices and CER scores must have same length"
        )
        assert len(self.real_indices) == len(self.real_cer_scores), (
            "Real indices and CER scores must have same length"
        )

        logger.info("CurriculumSampler initialized:")
        logger.info("  Synthetic samples: %d", len(self.synthetic_indices))
        logger.info("  Real samples: %d", len(self.real_indices))
        logger.info("  Initial synthetic ratio: %.3f", self.initial_synthetic_ratio)
        logger.info("  Final synthetic ratio: %.3f", self.final_synthetic_ratio)
        logger.info("  Balance by speaker: %s", self.balance_by_speaker)

        self.current_epoch = 0

    def __iter__(self):
        if not self.synthetic_indices and not self.real_indices:
            raise RuntimeError("CurriculumSampler has no data to sample from.")

        current_epoch = self.current_epoch if self.is_training else self.num_epochs
        # 1) synthetic_ratio 按 synthetic_ratio_start/end_epoch 调度
        if self.synthetic_ratio_end_epoch <= self.synthetic_ratio_start_epoch:
            ratio_progress = 1.0
        else:
            ratio_progress = (
                (current_epoch - self.synthetic_ratio_start_epoch)
                / max(1, self.synthetic_ratio_end_epoch - self.synthetic_ratio_start_epoch)
            )
            ratio_progress = min(1.0, max(0.0, ratio_progress))

        current_synthetic_ratio = (
            self.initial_synthetic_ratio * (1 - ratio_progress)
            + self.final_synthetic_ratio * ratio_progress
        )

        # 2) difficulty_threshold 仍然按全局 num_epochs 线性调度（保持原逻辑）
        progress = min(current_epoch / self.num_epochs, 1.0)
        current_difficulty_threshold = (
            self.initial_difficulty_threshold * (1 - progress)
            + self.final_difficulty_threshold * progress
        )

        synthetic_selected = self._select_by_strategy(
            self.synthetic_indices,
            self.synthetic_cer_scores,
            self.synthetic_speaker_indices,
            current_difficulty_threshold,
        )
        real_selected = self._select_by_strategy(
            self.real_indices,
            self.real_cer_scores,
            self.real_speaker_indices,
            current_difficulty_threshold,
        )

        indices = self._mix_data(
            synthetic_selected, real_selected, current_synthetic_ratio
        )

        if not indices:
            indices = list(range(len(self.dataset)))

        # 在“易→难”的基础上，按 epoch 逐渐加入一点随机噪声，让顺序
        # 从强课程慢慢过渡到更接近随机。
        if self.difficulty_noise_max > 0 and self.is_training:
            indices = self._apply_difficulty_noise(indices)

        if self.shuffle:
            random.shuffle(indices)

        batches = self._create_batches(indices)

        if not batches:
            batches = [indices]

        if not batches:
            batches = [indices]

        if self.current_epoch % 10 == 0 and self.is_training:
            logger.info(
                "Epoch %d: synthetic_ratio=%.3f, difficulty_threshold=%.3f, total_samples=%d",
                self.current_epoch,
                current_synthetic_ratio,
                current_difficulty_threshold,
                len(indices),
            )

        return iter(batches)

    def _apply_difficulty_noise(self, indices: List[int]) -> List[int]:
        """
        在难度顺序（CER）上添加 epoch 相关的随机噪声：
        - 前期：噪声接近 0，基本保持易→难
        - 中期：噪声逐渐增大，局部打乱
        - 后期：噪声接近 difficulty_noise_max，整体接近随机
        """
        if not indices:
            return indices

        # 计算当前 epoch 在噪声调度区间内的进度（0 ~ 1）
        if self.difficulty_noise_end_epoch <= self.difficulty_noise_start_epoch:
            progress = 1.0
        else:
            progress = (
                (self.current_epoch - self.difficulty_noise_start_epoch)
                / max(1, self.difficulty_noise_end_epoch - self.difficulty_noise_start_epoch)
            )
            progress = min(1.0, max(0.0, progress))

        if progress <= 0.0:
            # 尚未进入加噪声阶段，直接返回原顺序
            return indices

        noise_scale = self.difficulty_noise_max * progress
        if noise_scale <= 0.0:
            return indices

        # 构建 index -> cer 的查找表（合成+真实）
        cer_map: Dict[int, float] = {}
        for idx, cer in zip(self.synthetic_indices, self.synthetic_cer_scores):
            cer_map[idx] = float(cer)
        for idx, cer in zip(self.real_indices, self.real_cer_scores):
            cer_map[idx] = float(cer)

        # CER 可能缺失时，用一个中等难度兜底
        default_cer = 0.5

        def noisy_key(i: int) -> float:
            base = cer_map.get(i, default_cer)
            # 在 [-1, 1] 上采样一个随机数，加到 CER 上
            eps = random.uniform(-1.0, 1.0)
            return base + noise_scale * eps

        # 根据带噪声的“难度”重新排序
        return sorted(indices, key=noisy_key)

    def _create_batches(self, indices: List[int]) -> List[List[int]]:
        # 纯按样本个数打包
        if self.batch_type == "example":
            return self._simple_batch(indices)

        # 按长度约束打包：如果配置了 length_bucket_size，则先在局部窗口内
        # 做“长度友好”的排序，再调用原来的 length_constrained 逻辑。
        if self.length_bucket_size and self.length_bucket_size > 0:
            all_batches: List[List[int]] = []
            bucket_size = self.length_bucket_size

            for start in range(0, len(indices), bucket_size):
                window = indices[start : start + bucket_size]
                if not window:
                    continue

                # 在局部 CER 相近的窗口内，按长度从短到长排序，
                # 既能维持“分段易→难”，又能让 batch 更饱满。
                # 若取不到长度（返回 None），用一个中等长度兜底，避免丢样本。
                def _length_key(i: int) -> int:
                    l = self._get_sample_length(i)
                    if l is None:
                        return 1
                    return l

                window_sorted = sorted(window, key=_length_key)
                bucket_batches = self._length_constrained_batch(window_sorted)
                if bucket_batches:
                    all_batches.extend(bucket_batches)

            self._last_batch_count = len(all_batches)
            return all_batches

        # 默认：直接在全局 CER 顺序上做 length-constrained 打包
        return self._length_constrained_batch(indices)

    def _simple_batch(self, indices: List[int]) -> List[List[int]]:
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        self._last_batch_count = len(batches)
        return batches

    def _length_constrained_batch(self, indices: List[int]) -> List[List[int]]:
        batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            sample_len = self._get_sample_length(idx)
            if sample_len is None:
                continue

            potential = max(current_max_len, sample_len) * (len(current_batch) + 1)
            if potential <= self.batch_size:
                current_batch.append(idx)
                current_max_len = max(current_max_len, sample_len)
            else:
                if current_batch and (len(current_batch) == self.batch_size or not self.drop_last):
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len

        if current_batch and (len(current_batch) == self.batch_size or not self.drop_last):
            batches.append(current_batch)

        self._last_batch_count = len(batches)
        return batches

    def _get_sample_length(self, idx: int) -> Optional[int]:
        if self.batch_type == "length":
            length = self.dataset.get_source_len(idx)
        elif self.batch_type == "token":
            src_len = self.dataset.get_source_len(idx)
            tgt_len = 0
            if hasattr(self.dataset, "get_target_len"):
                try:
                    tgt_len = int(self.dataset.get_target_len(idx) * 1.2)
                except Exception:
                    tgt_len = 0
            length = src_len + tgt_len
        else:
            return 1

        if self.max_token_length and length > self.max_token_length:
            return None
        if self.min_token_length and length < self.min_token_length:
            return None
        return max(1, length)

    def _select_by_strategy(
        self,
        indices: List[int],
        cer_scores: List[float],
        speaker_indices_dict: Optional[Dict[str, List[int]]],
        threshold: float,
    ):
        if not indices:
            return []
        if self.balance_by_speaker and speaker_indices_dict:
            selected = []
            for speaker_id, spk_indices in speaker_indices_dict.items():
                num_select = (
                    int(len(spk_indices) * threshold)
                    if threshold < 1.0
                    else len(spk_indices)
                )
                num_select = max(1, num_select) if self.is_training else len(spk_indices)
                selected.extend(spk_indices[:num_select])
            return selected

        num_select = int(len(indices) * threshold) if threshold < 1.0 else len(indices)
        if self.is_training:
            num_select = max(1, num_select)
        return indices[:num_select]

    def _select_by_difficulty(self, indices, cer_scores, threshold):
        """根据难度阈值选择数据"""
        # threshold是0-1之间的值，表示选择前threshold比例的数据
        num_select = int(len(indices) * threshold) if threshold < 1.0 else len(indices)
        return indices[:num_select]
    
    def _select_by_difficulty_per_speaker(self, speaker_indices_dict, threshold):
        """
        按说话人均衡选择数据
        每个说话人都选择前threshold比例的数据
        """
        selected_indices = []
        for speaker_id, indices in speaker_indices_dict.items():
            num_select = int(len(indices) * threshold) if threshold < 1.0 else len(indices)
            selected_indices.extend(indices[:num_select])
        return selected_indices
    
    def _mix_data(self, synthetic_indices, real_indices, synthetic_ratio):
        """根据比例混合合成和真实数据"""
        total_synthetic = len(synthetic_indices)
        total_real = len(real_indices)
        
        if total_synthetic == 0 and total_real == 0:
            return []
        
        if total_synthetic == 0:
            return real_indices
        if total_real == 0:
            return synthetic_indices
        
        # 计算实际需要的数量（保持比例）
        # 假设总样本数为 min(len(synthetic), len(real)) * 2
        max_samples = min(total_synthetic + total_real, 
                         int(total_synthetic / synthetic_ratio) if synthetic_ratio > 0 else total_real)
        
        num_synthetic = int(max_samples * synthetic_ratio)
        num_real = max_samples - num_synthetic
        
        # 确保不超过实际数量
        num_synthetic = min(num_synthetic, total_synthetic)
        num_real = min(num_real, total_real)
        
        # 选择数据
        selected_synthetic = synthetic_indices[:num_synthetic]
        selected_real = real_indices[:num_real]
        
        # 混合（可以调整混合策略）
        mixed = selected_synthetic + selected_real
        
        return mixed
    
    def __len__(self):
        # 返回最大可能的长度
        return max(1, self._last_batch_count)
    
    def set_epoch(self, epoch):
        """设置当前epoch（用于分布式训练）"""
        self.current_epoch = epoch


@tables.register("batch_sampler_classes", "AdaptiveCurriculumSampler")
class AdaptiveCurriculumSampler(CurriculumSamplerCore):
    """
    自适应课程学习采样器
    
    根据模型在验证集上的表现动态调整难度
    """
    
    def __init__(
        self,
        dataset,
        synthetic_indices: List[int],
        real_indices: List[int],
        synthetic_cer_scores: List[float],
        real_cer_scores: List[float],
        num_epochs: int = 100,
        initial_synthetic_ratio: float = 1.0,
        final_synthetic_ratio: float = 0.3,
        initial_difficulty_threshold: float = 0.0,
        final_difficulty_threshold: float = 1.0,
        validation_cer_threshold: float = 0.15,  # 验证集CER低于此值时增加难度
        shuffle: bool = True,
        balance_by_speaker: bool = True,
        synthetic_speaker_indices: Dict[str, List[int]] = None,
        real_speaker_indices: Dict[str, List[int]] = None,
        **kwargs
    ):
        super().__init__(
            dataset, synthetic_indices, real_indices,
            synthetic_cer_scores, real_cer_scores,
            num_epochs, initial_synthetic_ratio, final_synthetic_ratio,
            initial_difficulty_threshold, final_difficulty_threshold,
            shuffle, balance_by_speaker=balance_by_speaker,
            synthetic_speaker_indices=synthetic_speaker_indices,
            real_speaker_indices=real_speaker_indices,
            **kwargs
        )
        self.validation_cer_threshold = validation_cer_threshold
        self.current_difficulty = initial_difficulty_threshold
        self.current_synthetic_ratio = initial_synthetic_ratio
    
    def update_from_validation(self, validation_cer: float):
        """
        根据验证集CER更新采样策略
        
        Args:
            validation_cer: 验证集上的CER
        """
        # 如果验证集CER低于阈值，增加难度
        if validation_cer < self.validation_cer_threshold:
            # 增加难度阈值（选择更难的样本）
            self.current_difficulty = min(
                self.current_difficulty + 0.1,
                self.final_difficulty_threshold
            )
            # 减少合成数据比例
            self.current_synthetic_ratio = max(
                self.current_synthetic_ratio - 0.05,
                self.final_synthetic_ratio
            )
            logging.info(
                f"Validation CER {validation_cer:.4f} < {self.validation_cer_threshold}, "
                f"increasing difficulty: {self.current_difficulty:.3f}, "
                f"synthetic_ratio: {self.current_synthetic_ratio:.3f}"
            )
    
    def __iter__(self):
        # 使用自适应参数而不是线性插值
        if self.balance_by_speaker and self.synthetic_speaker_indices:
            synthetic_selected = self._select_by_difficulty_per_speaker(
                self.synthetic_speaker_indices,
                self.current_difficulty
            )
        else:
            synthetic_selected = self._select_by_difficulty(
                self.synthetic_indices,
                self.synthetic_cer_scores,
                self.current_difficulty
            )
        
        if self.balance_by_speaker and self.real_speaker_indices:
            real_selected = self._select_by_difficulty_per_speaker(
                self.real_speaker_indices,
                self.current_difficulty
            )
        else:
            real_selected = self._select_by_difficulty(
                self.real_indices,
                self.real_cer_scores,
                self.current_difficulty
            )
        
        indices = self._mix_data(
            synthetic_selected,
            real_selected,
            self.current_synthetic_ratio
        )
        
        if self.shuffle:
            random.shuffle(indices)
        
        return iter(indices)

