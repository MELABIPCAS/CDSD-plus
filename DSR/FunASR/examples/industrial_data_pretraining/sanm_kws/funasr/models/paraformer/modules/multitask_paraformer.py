#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务学习版本的 Paraformer：同时学习 ASR 和说话人分类
用于提升说话人无关的准确率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from funasr.register import tables
from funasr.models.paraformer.modules.film_adapter import ParaformerSpeaker
from funasr.models.transformer.utils.nets_utils import make_pad_mask


@tables.register("model_classes", "ParaformerMultiTask")
class ParaformerMultiTask(ParaformerSpeaker):
    """
    多任务学习版本的 Paraformer：
    - 主任务：ASR（语音识别）
    - 辅助任务：说话人分类（帮助模型学习说话人无关特征）
    
    通过同时学习两个任务，模型会学习到更通用的特征表示，
    从而提升在未见过的说话人上的表现。
    """

    def __init__(
        self,
        num_speakers: int = 100,  # 说话人总数
        speaker_weight: float = 0.0,  # 说话人分类损失的权重（设为0表示不使用说话人分类任务）
        speaker_hidden_size: int = 256,  # 说话人分类头的隐藏层大小
        speaker_dropout: float = 0.1,  # 说话人分类头的 dropout
        use_speaker_classification: bool = False,  # 是否使用说话人分类任务（默认False，只使用说话人嵌入作为条件）
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.num_speakers = num_speakers
        self.speaker_weight = speaker_weight
        self.speaker_hidden_size = speaker_hidden_size
        self.use_speaker_classification = use_speaker_classification
        
        # 说话人分类头：从 encoder 输出预测说话人 ID
        # 注意：如果use_speaker_classification=False，这些组件不会被使用
        # 但为了兼容性，仍然创建它们（如果speaker_weight>0，则使用）
        encoder_output_size = self.encoder.output_size()
        
        # 使用平均池化 + 全连接层进行分类
        self.speaker_classifier = nn.Sequential(
            nn.Linear(encoder_output_size, speaker_hidden_size),
            nn.ReLU(),
            nn.Dropout(speaker_dropout),
            nn.Linear(speaker_hidden_size, num_speakers),
        )
        
        # 说话人分类损失（使用 ignore_index=-1 来处理未见过的说话人）
        self.speaker_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # 用于标记哪些说话人ID是训练集中见过的（用于验证/测试时的过滤）
        self.seen_speaker_ids = None
        
        # 确保_current_speaker_ids被初始化（从父类继承，但确保存在）
        if not hasattr(self, "_current_speaker_ids"):
            self._current_speaker_ids = None
        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with multi-task learning
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            speaker_ids: (Batch,) - 说话人 ID，用于多任务学习
        """
        # 多任务学习：说话人分类（仅在训练时，或验证/测试时说话人在训练集中）
        # 注意：我们需要在调用父类forward之前准备好speaker_ids，确保训练和推理一致
        speaker_ids_for_asr = None
        speaker_ids_orig = None
        valid_speaker_mask = None
        
        if speaker_ids is not None:
            # 规范形状到 (B,)
            speaker_ids_orig = speaker_ids.to(torch.long)
            if speaker_ids_orig.dim() > 1:
                speaker_ids_orig = speaker_ids_orig.view(-1)
            
            # 处理未见过的说话人：将超出范围的 speaker_id 设为 -1（ignore_index）
            # 训练集中只有 num_speakers 个说话人，验证/测试集中可能有未见过的说话人
            # 注意：在传递给encode之前，需要clamp到合法范围，但保留原始ID用于损失计算
            valid_speaker_mask = (speaker_ids_orig >= 0) & (speaker_ids_orig < self.num_speakers)
            
            # 对于encode，使用clamp后的ID（避免索引越界）
            speaker_ids_for_asr = torch.clamp(speaker_ids_orig, min=0, max=self.num_speakers - 1)
        
        # 调用父类的 forward 进行 ASR 主任务
        # 父类会调用 encode，我们在这里缓存 encoder_out 用于说话人分类
        loss_asr, stats, weight = super().forward(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            speaker_ids=speaker_ids_for_asr,  # 使用处理后的speaker_ids
            **kwargs,
        )
        
        # 多任务学习：说话人分类（可选）
        # 注意：说话人嵌入已经通过FiLM注入到encoder作为条件，通常不需要额外的分类任务
        # 如果使用说话人分类任务，可能导致训练和推理不一致（训练时依赖分类任务，推理时没有）
        # 推荐：将speaker_weight设为0或use_speaker_classification设为False，只使用说话人嵌入作为条件
        if self.use_speaker_classification and self.speaker_weight > 0 and speaker_ids is not None and speaker_ids_orig is not None:
            # 方案1：使用encoder的早期层特征（在FiLM调制之前）
            # 需要修改encoder以返回中间层特征，比较复杂
            
            # 方案2：使用encoder的最终输出，但添加正则化防止"作弊"
            # 当前实现：使用最终输出，但通过dropout和正则化来防止过拟合
            # 如果loss_speaker很快降到0，说明模型可能"作弊"了，需要调整speaker_weight或使用方案1
            
            # 获取 encoder 输出（用于说话人分类）
            # 注意：这里会再次调用encode，但使用相同的speaker_ids，应该得到相同的结果
            encoder_out, encoder_out_lens = self.encode(
                speech, speech_lengths, speaker_ids=speaker_ids_for_asr
            )
            
            # 平均池化：对每个样本的 encoder 输出进行时间维度的平均
            # encoder_out: (Batch, Time, Dim)
            # 使用 mask 进行有效长度的平均
            masks = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))).to(encoder_out.device)  # (B, T)
            # 扩展到 (B, T, 1) 以匹配 encoder_out (B, T, D)
            masked_encoder_out = encoder_out * masks.float().unsqueeze(-1)
            # 对时间维度求平均（只考虑有效长度）
            pooled_features = masked_encoder_out.sum(dim=1) / encoder_out_lens.unsqueeze(1).float()
            # pooled_features: (Batch, Dim)
            
            # 说话人分类
            # 注意：如果loss_speaker很快降到0且acc_speaker=1.0，说明模型可能"作弊"了
            # 可以考虑：
            # 1. 降低speaker_weight（减少说话人分类任务的影响）
            # 2. 使用encoder的早期层特征（需要修改encoder返回中间层特征）
            # 3. 添加对抗训练（让encoder学习说话人无关的特征）
            speaker_logits = self.speaker_classifier(pooled_features)
            # speaker_logits: (Batch, num_speakers)
            
            # 使用原始ID计算损失（超出范围的设为-1）
            speaker_ids_masked = speaker_ids_orig.clone()
            speaker_ids_masked[~valid_speaker_mask] = -1  # ignore_index
            
            # 只在有有效说话人ID的样本上计算损失
            if valid_speaker_mask.any():
                # 计算说话人分类损失（自动忽略 speaker_id=-1 的样本）
                loss_speaker = self.speaker_criterion(speaker_logits, speaker_ids_masked)
                
                # 计算说话人分类准确率（只统计有效样本）
                speaker_pred = speaker_logits.argmax(dim=-1)
                valid_acc = (speaker_pred == speaker_ids_orig) & valid_speaker_mask
                speaker_acc = valid_acc.float().sum() / valid_speaker_mask.float().sum() if valid_speaker_mask.any() else torch.tensor(0.0)
                
                # 更新 stats
                stats["loss_speaker"] = loss_speaker.item()
                stats["acc_speaker"] = speaker_acc.item()
                stats["num_valid_speakers"] = valid_speaker_mask.sum().item()
                stats["num_ood_speakers"] = (~valid_speaker_mask).sum().item()  # Out-of-distribution speakers
                
                # 只在训练时或所有说话人都在训练集中时，将说话人分类损失加入总损失
                # 注意：为了确保训练和推理一致，建议只在训练时加入说话人分类损失
                if self.training:
                    total_loss = loss_asr + self.speaker_weight * loss_speaker
                    return total_loss, stats, weight
                else:
                    # 验证/测试时，即使有说话人分类任务，也不加入损失（确保与推理一致）
                    return loss_asr, stats, weight
            else:
                # 所有说话人都是未见过的（验证/测试集）
                stats["loss_speaker"] = 0.0
                stats["acc_speaker"] = 0.0
                stats["num_valid_speakers"] = 0
                stats["num_ood_speakers"] = speaker_ids_orig.size(0)
                # 只返回 ASR 损失
                return loss_asr, stats, weight
        
        # 如果没有使用说话人分类任务，或没有speaker_ids，只返回 ASR 损失
        return loss_asr, stats, weight
    
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """
        推理时只进行 ASR，不进行说话人分类
        使用标准接口以兼容 auto_model.py
        从 key 中提取 speaker_ids 并传递给 encode
        """
        import logging
        logging.info("开始推理，准备提取说话人ID...")
        
        # 从 key 中提取 speaker_ids（如果 key 存在）
        # speaker_id_prefix 从 dataset_conf 或 kwargs 中获取，默认为 4
        speaker_id_prefix = kwargs.get("speaker_id_prefix", 4)
        
        # 确保_current_speaker_ids被初始化
        if not hasattr(self, "_current_speaker_ids"):
            self._current_speaker_ids = None
        
        # 如果 key 存在，从 key 中提取 speaker_ids
        if key is not None and len(key) > 0:
            # 获取 speaker2idx 映射（优先从 dataset 获取，与训练时一致）
            speaker2idx = {}
            
            # 方法1: 从 dataset_conf 中获取 speaker_table_path，然后加载（与训练时 AudioDatasetSpeaker 的逻辑一致）
            dataset_conf = kwargs.get("dataset_conf", {})
            if isinstance(dataset_conf, dict):
                speaker_table_path = dataset_conf.get("speaker_table_path")
                if speaker_table_path:
                    try:
                        data = torch.load(speaker_table_path, map_location="cpu")
                        speaker2idx = data.get("speaker2idx", {})
                        import logging
                        logging.info(f"[推理] 从 dataset_conf.speaker_table_path={speaker_table_path} 加载，共 {len(speaker2idx)} 个说话人")
                    except Exception as e:
                        import logging
                        logging.warning(f"[推理] 无法从 dataset_conf.speaker_table_path 加载 speaker2idx: {e}")
            
            # 方法2: 从 encoder 的 speaker_encoder 中获取（最可靠的方法）
            if not speaker2idx and hasattr(self.encoder, "speaker_encoder"):
                se = self.encoder.speaker_encoder
                if hasattr(se, "speaker2idx"):
                    speaker2idx = se.speaker2idx
                    import logging
                    logging.info(f"[推理] 从 encoder.speaker_encoder.speaker2idx 获取，共 {len(speaker2idx)} 个说话人")
                elif hasattr(se, "spk_table_info"):
                    speaker2idx = se.spk_table_info.get("speaker2idx", {})
                    import logging
                    logging.info(f"[推理] 从 encoder.speaker_encoder.spk_table_info 获取，共 {len(speaker2idx)} 个说话人")
                # 如果speaker_encoder有table_path，尝试加载
                if not speaker2idx and hasattr(se, "table_path") and se.table_path:
                    try:
                        data = torch.load(se.table_path, map_location="cpu")
                        speaker2idx = data.get("speaker2idx", {})
                    except Exception as e:
                        import logging
                        logging.warning(f"[推理] 无法从 speaker_encoder.table_path 加载 speaker2idx: {e}")
            
            # 方法3: 从 encoder_conf.speaker_encoder_conf.table_path 获取（Hydra配置）
            if not speaker2idx:
                table_path = None
                # 从 kwargs 的嵌套配置中获取（Hydra 会将嵌套配置展开为点号分隔的键）
                for possible_key in ["encoder_conf.speaker_encoder_conf.table_path", 
                                    "speaker_encoder_conf.table_path",
                                    "table_path"]:
                    if possible_key in kwargs:
                        table_path = kwargs[possible_key]
                        break
                # 从 encoder_conf 中获取
                if not table_path:
                    encoder_conf = kwargs.get("encoder_conf", {})
                    if isinstance(encoder_conf, dict):
                        speaker_encoder_conf = encoder_conf.get("speaker_encoder_conf", {})
                        if isinstance(speaker_encoder_conf, dict):
                            table_path = speaker_encoder_conf.get("table_path")
                # 从 encoder 的属性中获取
                if not table_path and hasattr(self.encoder, "speaker_encoder_conf"):
                    table_path = self.encoder.speaker_encoder_conf.get("table_path")
                # 从 kwargs 中直接获取
                if not table_path:
                    table_path = kwargs.get("table_path")
                
                if table_path:
                    try:
                        data = torch.load(table_path, map_location="cpu")
                        speaker2idx = data.get("speaker2idx", {})
                        import logging
                        logging.info(f"成功从 {table_path} 加载 speaker2idx，共 {len(speaker2idx)} 个说话人")
                    except Exception as e:
                        import logging
                        logging.warning(f"无法加载 speaker2idx 从 {table_path}: {e}")
                        pass
            
            # 如果仍然没有 speaker2idx，发出警告并尝试从encoder直接获取
            if not speaker2idx:
                import logging
                # 最后尝试：直接从encoder的配置中获取
                if hasattr(self.encoder, "speaker_encoder"):
                    se = self.encoder.speaker_encoder
                    # 检查是否有table_path属性
                    if hasattr(se, "table_path") and se.table_path:
                        try:
                            data = torch.load(se.table_path, map_location="cpu")
                            speaker2idx = data.get("speaker2idx", {})
                            logging.info(f"从 encoder.speaker_encoder.table_path 加载 speaker2idx，共 {len(speaker2idx)} 个说话人")
                        except Exception as e:
                            logging.warning(f"无法从 encoder.speaker_encoder.table_path 加载: {e}")
                
                if not speaker2idx:
                    logging.warning("推理时无法获取 speaker2idx，所有样本将使用默认说话人ID 0")
                    logging.warning("请确保在推理时传递 encoder_conf.speaker_encoder_conf.table_path 或 dataset_conf.speaker_table_path")
            
            # 从 key 中提取 speaker_id
            speaker_ids_list = []
            unknown_count = 0
            for k in key:
                if isinstance(k, (list, tuple)):
                    k = k[0] if len(k) > 0 else ""
                speaker_id_str = str(k)[:speaker_id_prefix] if k else ""
                # 如果找不到，尝试使用 speaker_id_str 本身作为索引（如果它是数字）
                if speaker_id_str in speaker2idx:
                    speaker_idx = speaker2idx[speaker_id_str]
                else:
                    # 如果表中没有，使用 0（默认说话人）
                    # 注意：这可能导致未知说话人使用相同的嵌入，但比索引越界好
                    speaker_idx = 0
                    unknown_count += 1
                speaker_ids_list.append(speaker_idx)
            
            # 如果有很多未知说话人，发出警告
            if unknown_count > 0 and len(key) > 0:
                import logging
                logging.warning(f"推理时发现 {unknown_count}/{len(key)} 个未知说话人ID，将使用默认说话人嵌入（ID=0）")
            
            # 转换为 tensor
            if speaker_ids_list:
                speaker_ids = torch.tensor(speaker_ids_list, dtype=torch.long)
                # 设置到 _current_speaker_ids，这样 encode() 可以获取到
                self._current_speaker_ids = speaker_ids
                # 同时通过 kwargs 传递，确保 encode() 能获取到
                kwargs["speaker_ids"] = speaker_ids
                # 添加调试信息
                unique_speaker_ids = torch.unique(speaker_ids).tolist()
        else:
            # 如果 key 不存在，设置默认值避免错误
            if self._current_speaker_ids is None:
                # 创建一个默认的 speaker_ids（batch_size=1）
                self._current_speaker_ids = torch.tensor([0], dtype=torch.long)
                kwargs["speaker_ids"] = self._current_speaker_ids
        
        # 重写 encode 方法，确保传递 speaker_ids 并移动到正确设备
        original_encode = self.encode
        def encode_with_speaker_ids(speech, speech_lengths, **encode_kwargs):
            # 优先从encode_kwargs获取，如果没有则从self._current_speaker_ids获取
            speaker_ids = encode_kwargs.get("speaker_ids", None)
            if speaker_ids is None:
                speaker_ids = self._current_speaker_ids
            
            if speaker_ids is not None:
                # 确保 batch size 匹配
                batch_size = speech.size(0) if speech.dim() > 1 else 1
                if isinstance(speaker_ids, torch.Tensor):
                    if speaker_ids.size(0) != batch_size:
                        # 如果 batch size 不匹配，使用第一个 speaker_id 或截取/填充
                        if batch_size > speaker_ids.size(0):
                            # 如果需要的 batch size 更大，使用第一个 speaker_id 填充
                            first_id = speaker_ids[0:1]
                            speaker_ids = first_id.repeat(batch_size)
                        else:
                            # 如果需要的 batch size 更小，截取
                            speaker_ids = speaker_ids[:batch_size]
                    # 移动到与 speech 相同的设备
                    speaker_ids = speaker_ids.to(speech.device)
                else:
                    # 如果不是tensor，转换为tensor
                    if not isinstance(speaker_ids, (list, tuple)):
                        speaker_ids = [speaker_ids]
                    speaker_ids = torch.tensor(speaker_ids[:batch_size], dtype=torch.long, device=speech.device)
                
                # 确保speaker_ids被传递到encode_kwargs
                encode_kwargs["speaker_ids"] = speaker_ids
                
                # 调试：检查speaker_ids是否被正确传递（使用INFO级别确保输出）
                import logging
                if speaker_ids.numel() > 0:
                    unique_ids = torch.unique(speaker_ids).tolist()
                else:
                    logging.warning(f"[推理] encode调用时speaker_ids为空！")
            else:
                import logging
                logging.warning("[推理] encode调用时没有speaker_ids，将使用默认值或None")
                # 即使没有speaker_ids，也确保传递None，让encoder知道
                encode_kwargs["speaker_ids"] = None
            
            return original_encode(speech, speech_lengths, **encode_kwargs)
        
        # 临时替换 encode 方法，确保speaker_ids被正确传递
        self.encode = encode_with_speaker_ids
        # 同时设置_current_speaker_ids，确保父类的inference方法也能获取到
        if hasattr(self, "_current_speaker_ids") and self._current_speaker_ids is not None:
            # 确保_current_speaker_ids被设置（已经在上面设置了）
            pass
        
        try:
            # 调用父类的inference，它会调用encode，而encode会从_current_speaker_ids或kwargs中获取speaker_ids
            result = super().inference(
                data_in=data_in,
                data_lengths=data_lengths,
                key=key,
                tokenizer=tokenizer,
                frontend=frontend,
                **kwargs,
            )
            return result
        finally:
            # 恢复原始 encode 方法
            self.encode = original_encode
            # 清理_current_speaker_ids（可选，但为了安全）
            # self._current_speaker_ids = None  # 不清理，可能后续还需要



