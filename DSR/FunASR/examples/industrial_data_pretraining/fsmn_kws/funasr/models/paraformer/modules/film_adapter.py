#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from funasr.register import tables
from funasr.models.sanm.encoder import SANMEncoder
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.sanm.encoder import TooShortUttError, check_short_utt
from funasr.models.speaker.ecapa_table_encoder import ECAPATableEncoder
try:
    from funasr.models.speaker.cross_attention_speaker_encoder import CrossAttentionSpeakerEncoder
    _HAS_CROSS_ATTENTION = True
except ImportError:
    CrossAttentionSpeakerEncoder = None
    _HAS_CROSS_ATTENTION = False
from funasr.models.paraformer.model import Paraformer


class FiLMAffine(nn.Module):
    """将 speaker embedding 映射成 γ/β。"""

    def __init__(self, embed_dim: int, hidden_dim: int, target_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim * 2),
        )

    def forward(self, spk_embed: torch.Tensor):
        params = self.net(spk_embed)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta


class FiLMLayerWrapper(nn.Module):
    """包裹 EncoderLayerSANM，在输出处注入 FiLM。"""

    def __init__(self, layer: nn.Module, film: FiLMAffine):
        super().__init__()
        self.layer = layer
        self.film = film

    def forward(
        self,
        x,
        mask,
        cache=None,
        mask_shfit_chunk=None,
        mask_att_chunk_encoder=None,
        speaker_embed: Optional[torch.Tensor] = None,
    ):
        x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder = self.layer(
            x,
            mask,
            cache=cache,
            mask_shfit_chunk=mask_shfit_chunk,
            mask_att_chunk_encoder=mask_att_chunk_encoder,
        )
        if speaker_embed is not None:
            gamma, beta = self.film(speaker_embed)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = x * (1 + gamma) + beta
        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


@tables.register("encoder_classes", "SANMEncoderSpk")
class SANMEncoderSpk(SANMEncoder):
    """支持 FiLM 说话人调制的 SANM encoder。"""

    def __init__(
        self,
        *args,
        use_spk_condition: bool = True,
        film_hidden_dim: int = 256,
        film_dropout: float = 0.1,
        speaker_encoder_conf: Optional[dict] = None,
        **kwargs,
    ):
        self.use_spk_condition = use_spk_condition
        self.film_hidden_dim = film_hidden_dim
        self.film_dropout = film_dropout
        # OmegaConf 可能是只读 DictConfig，这里转成普通 dict 避免 pop 报错
        from omegaconf import OmegaConf
        if speaker_encoder_conf is None:
            self.speaker_encoder_conf = {}
        elif isinstance(speaker_encoder_conf, dict):
            self.speaker_encoder_conf = dict(speaker_encoder_conf)
        else:
            self.speaker_encoder_conf = OmegaConf.to_container(
                speaker_encoder_conf, resolve=True, enum_to_str=True
            )
        super().__init__(*args, **kwargs)

        self.speaker_encoder = None
        self.film_layers = None
        # 解析 Cross-Attention 开关并移除仅供 CA 使用的参数，防止传给 ECAPATableEncoder 报错
        self.use_cross_attention = bool(self.speaker_encoder_conf.pop("use_cross_attention", False))
        ca_kwargs = {}
        for k in ["use_length_weight", "use_uncertainty", "min_length_threshold"]:
            if k in self.speaker_encoder_conf:
                ca_kwargs[k] = self.speaker_encoder_conf.pop(k)
        if self.use_spk_condition:
            table_path = self.speaker_encoder_conf.get("table_path")
            if not table_path:
                raise ValueError("speaker_encoder_conf.table_path 必须提供")
            
            # 选择使用 Cross-Attention 聚合还是简单的查表
            if self.use_cross_attention and _HAS_CROSS_ATTENTION:
                try:
                    self.speaker_encoder = CrossAttentionSpeakerEncoder(
                        **self.speaker_encoder_conf, **ca_kwargs
                    )
                    embed_dim = self.speaker_encoder.embed_dim
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"无法初始化 CrossAttentionSpeakerEncoder: {e}，回退到 ECAPATableEncoder"
                    )
                    self.speaker_encoder = ECAPATableEncoder(**self.speaker_encoder_conf)
                    embed_dim = self.speaker_encoder.embedding.embedding_dim
                    self.use_cross_attention = False
            else:
                self.speaker_encoder = ECAPATableEncoder(**self.speaker_encoder_conf)
                embed_dim = self.speaker_encoder.embedding.embedding_dim
            total_layers = len(self.encoders0) + len(self.encoders)
            self.film_layers = nn.ModuleList(
                [
                    FiLMAffine(embed_dim, self.film_hidden_dim, self._output_size, self.film_dropout)
                    for _ in range(total_layers)
                ]
            )

    def _sanitize_speaker_ids(self, speaker_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        防止说话人 ID 越界或为负，避免 embedding lookup 触发 CUDA 断言。
        将 ID clamp 到 [0, num_embeddings-1]。
        """
        if speaker_ids is None:
            return None
        speaker_ids = speaker_ids.to(torch.long)
        num_embeddings = None
        if self.speaker_encoder is not None:
            # 支持 ECAPA table encoder
            if hasattr(self.speaker_encoder, "embedding"):
                num_embeddings = self.speaker_encoder.embedding.num_embeddings
            # 支持 CrossAttentionSpeakerEncoder (base_embedding)
            elif hasattr(self.speaker_encoder, "base_embedding"):
                num_embeddings = self.speaker_encoder.base_embedding.num_embeddings
        if num_embeddings is not None and num_embeddings > 0:
            speaker_ids = torch.clamp(speaker_ids, min=0, max=num_embeddings - 1)
        return speaker_ids

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.use_spk_condition:
            return super().forward(xs_pad, ilens, **kwargs)
        if speaker_ids is None:
            raise ValueError("启用 FiLM 时必须提供 speaker_ids")

        # 先清洗 speaker_ids，防止越界
        speaker_ids = self._sanitize_speaker_ids(speaker_ids)
        
        # 如果使用 Cross-Attention，传入音频特征和长度用于聚合
        # 注意：xs_pad 此时还未经过 embedding，需要先处理
        if self.use_cross_attention and isinstance(self.speaker_encoder, CrossAttentionSpeakerEncoder):
            # 先获取基础嵌入
            base_emb = self.speaker_encoder.base_embedding(speaker_ids)
            # 传入音频特征用于 Cross-Attention 聚合（使用embedding后的特征）
            # 但此时xs_pad还未经过embedding，所以先传入None，后续在embedding后再调用
            # 为了简化，我们在这里先使用基础嵌入，然后在embedding后再更新
            speaker_embed = base_emb
        else:
            # 简单的查表方式
            speaker_embed = self.speaker_encoder(speaker_ids)

        # 确保长度合法且为一维 long tensor，在 CPU 上构造 mask，避免 CUDA 断言/类型问题
        if not torch.is_tensor(ilens):
            ilens = torch.as_tensor(ilens, dtype=torch.long, device=xs_pad.device)
        else:
            ilens = ilens.to(torch.long)
        ilens = torch.clamp(ilens, min=1)
        if ilens.dim() > 1:
            ilens = ilens.view(-1)
        ilens_cpu = ilens.cpu()
        masks = (~make_pad_mask(ilens_cpu)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size() ** 0.5
        
        # 保存原始长度，用于后续计算
        original_ilens = ilens.clone() if isinstance(ilens, torch.Tensor) else torch.tensor(ilens, device=xs_pad.device, dtype=torch.long)
        
        if self.embed is None:
            xs_pad = xs_pad
            # 如果没有embedding，长度不变
            current_ilens = original_ilens
        elif isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8),
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
            # subsampling后，masks已经被更新，使用masks计算实际长度
            # masks: (batch, 1, time_new)，True表示有效位置
            if masks is not None and masks.numel() > 0:
                current_ilens = masks.squeeze(1).sum(dim=1).to(torch.long)  # (batch,)
                # 确保长度至少为1
                current_ilens = torch.clamp(current_ilens, min=1)
            else:
                # 如果masks为空，使用xs_pad的时间维度（假设所有样本都是满长度）
                current_ilens = torch.full((xs_pad.size(0),), xs_pad.size(1), dtype=torch.long, device=xs_pad.device)
        else:
            xs_pad = self.embed(xs_pad)
            # 如果embedding没有返回masks，需要重新计算
            # 对于非subsampling的embedding，长度通常不变，但需要检查
            # 如果masks的时间维度匹配xs_pad，使用masks；否则使用原始长度
            if masks is not None and masks.dim() > 0 and masks.size(-1) == xs_pad.size(1):
                current_ilens = masks.squeeze(1).sum(dim=1).to(torch.long)  # (batch,)
                current_ilens = torch.clamp(current_ilens, min=1)
            else:
                # 如果时间维度不匹配，假设embedding没有改变长度（或按比例缩放）
                # 保守起见，使用原始长度
                current_ilens = original_ilens
                current_ilens = torch.clamp(current_ilens, min=1)
        
        # 如果使用 Cross-Attention，在embedding后使用音频特征聚合说话人嵌入
        if self.use_cross_attention and _HAS_CROSS_ATTENTION and hasattr(self.speaker_encoder, 'compute_uncertainty_weight'):
            # 使用embedding后的特征来聚合说话人嵌入
            # 确保长度是有效的（>0）
            import logging
            if not self.training:  # 只在推理时输出详细日志，训练时减少日志
                logging.info(f"[推理] 使用Cross-Attention聚合说话人嵌入: batch_size={xs_pad.size(0)}, audio_features_shape={xs_pad.shape}, audio_lengths={current_ilens.tolist()[:5] if current_ilens.numel() > 0 else '[]'}")
            speaker_embed = self.speaker_encoder(
                speaker_ids,
                audio_features=xs_pad,  # (batch, time, dim) embedding后的特征
                audio_lengths=current_ilens,  # (batch,) embedding后的实际长度
            )
            if not self.training:
                logging.info(f"[推理] Cross-Attention聚合完成: speaker_embed_shape={speaker_embed.shape}")
        elif self.use_cross_attention:
            import logging
            if not self.training:
                logging.warning(f"[推理] Cross-Attention未启用: use_cross_attention={self.use_cross_attention}, _HAS_CROSS_ATTENTION={_HAS_CROSS_ATTENTION}, has_compute_uncertainty_weight={hasattr(self.speaker_encoder, 'compute_uncertainty_weight') if self.speaker_encoder else False}")

        # 手工迭代层，逐层应用 FiLM，而不改变原始层命名（保持 ckpt 对齐）
        layer_idx = 0
        for layer in self.encoders0:
            xs_pad, masks, _, _, _ = layer(xs_pad, masks, None, None, None)
            if speaker_embed is not None:
                gamma, beta = self.film_layers[layer_idx](speaker_embed)
                xs_pad = xs_pad * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            layer_idx += 1

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer in self.encoders:
                xs_pad, masks, _, _, _ = layer(xs_pad, masks, None, None, None)
                if speaker_embed is not None:
                    gamma, beta = self.film_layers[layer_idx](speaker_embed)
                    xs_pad = xs_pad * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
                layer_idx += 1
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks, _, _, _ = encoder_layer(xs_pad, masks, None, None, None)
                if speaker_embed is not None:
                    gamma, beta = self.film_layers[layer_idx](speaker_embed)
                    xs_pad = xs_pad * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
                layer_idx += 1
                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)
                    intermediate_outs.append((layer_idx + 1, encoder_out))

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None


@tables.register("model_classes", "ParaformerSpeaker")
class ParaformerSpeaker(Paraformer):
    """在 Paraformer 基础上注入 speaker_ids -> encoder。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_speaker_ids = None

    def _sanitize_speaker_ids(self, speaker_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        防止说话人 ID 为负或越界，统一转为 long 并 clamp 到合法范围。
        """
        if speaker_ids is None:
            return None
        # 形状标准化到 (B,) 避免后续 embedding 查询产生 4D query
        speaker_ids = speaker_ids.to(torch.long)
        if speaker_ids.dim() > 1:
            speaker_ids = speaker_ids.squeeze(-1)
        if speaker_ids.dim() > 1:
            speaker_ids = speaker_ids.view(-1)
        if speaker_ids.dim() == 0:
            speaker_ids = speaker_ids.unsqueeze(0)
        num_embeddings = None
        enc = getattr(self, "encoder", None)
        if enc is not None and hasattr(enc, "speaker_encoder"):
            se = enc.speaker_encoder
            if hasattr(se, "embedding"):
                num_embeddings = se.embedding.num_embeddings
            elif hasattr(se, "base_embedding"):
                num_embeddings = se.base_embedding.num_embeddings
        if num_embeddings is not None and num_embeddings > 0:
            speaker_ids = torch.clamp(speaker_ids, min=0, max=num_embeddings - 1)
        return speaker_ids

    def forward(self, *args, speaker_ids=None, **kwargs):
        self._current_speaker_ids = speaker_ids
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._current_speaker_ids = None

    def encode(self, speech, speech_lengths, **kwargs):
        speaker_ids = kwargs.get("speaker_ids", None)
        if speaker_ids is None:
            speaker_ids = self._current_speaker_ids
        # 清洗 speaker_ids，防止越界
        speaker_ids = self._sanitize_speaker_ids(speaker_ids)

        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        encoder_out, encoder_out_lens, _ = self.encoder(
            speech, speech_lengths, speaker_ids=speaker_ids
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return encoder_out, encoder_out_lens

