# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Modified: 添加前缀引导解码功能，支持 BiTransformerDecoder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")
    
    # ============== 新增：前缀引导参数 ==============
    parser.add_argument('--prefix_file',
                        type=str,
                        default='',
                        help='前缀文件路径，格式: key prefix_text (每行)')
    parser.add_argument('--prefix_mode',
                        type=str,
                        default='first_char',
                        choices=['first_char', 'full', 'n_chars'],
                        help='前缀模式: first_char=只用第一个字符, full=使用完整前缀, n_chars=使用前n个字符')
    parser.add_argument('--prefix_n_chars',
                        type=int,
                        default=1,
                        help='使用前缀的前n个字符 (当prefix_mode=n_chars时生效)')
    # =============================================
    
    args = parser.parse_args()
    print(args)
    return args


def load_prefix_file(prefix_file):
    """
    加载前缀文件
    """
    prefix_dict = {}
    if not prefix_file or not os.path.exists(prefix_file):
        return prefix_dict
    
    with open(prefix_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, text = parts
                prefix_dict[key] = text
    
    logging.info(f"加载了 {len(prefix_dict)} 条前缀")
    return prefix_dict


def get_prefix_tokens(text, tokenizer, mode='first_char', n_chars=1):
    """
    从文本中提取前缀并转换为 token ids
    """
    if not text:
        return []
    
    text = text.replace(' ', '').strip()
    
    if mode == 'first_char':
        prefix_text = text[0] if text else ''
    elif mode == 'n_chars':
        prefix_text = text[:n_chars] if text else ''
    else:
        prefix_text = text
    
    if not prefix_text:
        return []
    
    # 使用 tokenizer 转换
    if hasattr(tokenizer, 'text2tokens'):
        tokens = tokenizer.text2tokens(prefix_text)
        if hasattr(tokenizer, 'tokens2ids'):
            prefix_ids = tokenizer.tokens2ids(tokens)
        else:
            prefix_ids = [tokenizer.symbol_table.get(t, 0) for t in tokens]
    elif hasattr(tokenizer, 'tokenize'):
        prefix_ids = tokenizer.tokenize(prefix_text)
    elif hasattr(tokenizer, 'encode'):
        prefix_ids = tokenizer.encode(prefix_text)
    else:
        prefix_ids = []
        symbol_table = tokenizer.symbol_table
        for char in prefix_text:
            if char in symbol_table:
                prefix_ids.append(symbol_table[char])
    
    return prefix_ids


def subsequent_mask(size, device):
    """生成下三角 mask"""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0)


def beam_search_with_prefix(model, encoder_out, encoder_mask, beam_size,
                            prefix_ids, sos_id, eos_id, device,
                            reverse_weight=0.0, max_len=100):
    """
    带前缀的 Beam Search 实现 - 支持 BiTransformerDecoder
    
    Args:
        model: WeNet ASR 模型
        encoder_out: [1, time, dim] encoder 输出
        encoder_mask: [1, 1, time] encoder mask
        beam_size: beam 大小
        prefix_ids: [sos, prefix_1, prefix_2, ...] 前缀序列 (含 sos)
        sos_id: sos token id
        eos_id: eos token id
        device: 设备
        reverse_weight: 反向解码权重 (用于 BiTransformerDecoder)
        max_len: 最大解码长度
    
    Returns:
        List[DecodeResult]: 解码结果
    """
    try:
        from wenet.transformer.search import DecodeResult
    except ImportError:
        # 如果导入失败，定义一个简单的结果类
        class DecodeResult:
            def __init__(self, tokens, score):
                self.tokens = tokens
                self.score = score
    
    batch_size = encoder_out.size(0)
    assert batch_size == 1, "前缀解码目前只支持 batch_size=1"
    
    # 检查是否是 BiTransformerDecoder
    decoder = model.decoder
    is_bidirectional = 'BiTransformer' in decoder.__class__.__name__
    logging.debug(f"Decoder 类型: {decoder.__class__.__name__}, 双向: {is_bidirectional}")
    
    # 扩展 encoder_out 到 beam_size
    encoder_out = encoder_out.repeat(beam_size, 1, 1)
    encoder_mask = encoder_mask.repeat(beam_size, 1, 1)
    
    # 初始化
    prefix_len = len(prefix_ids)
    hyps = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    hyps = hyps.repeat(beam_size, 1)  # [beam, prefix_len]
    
    scores = torch.zeros(beam_size, device=device)
    ended_hyps = []
    
    # Beam search 主循环
    for step in range(prefix_len, max_len):
        cur_len = hyps.size(1)
        
        # 构建 mask
        tgt_mask = subsequent_mask(cur_len, device)
        
        # 构建输入长度
        ys_in_lens = torch.tensor([cur_len] * beam_size, device=device, dtype=torch.long)
        
        # 调用 decoder
        if is_bidirectional:
            # BiTransformerDecoder 需要反向序列
            # 反向序列: 将 hyps 反转 (不含 sos，加 sos 在末尾)
            # 例如: [sos, a, b, c] -> [c, b, a, sos]
            r_hyps = torch.flip(hyps, dims=[1])
            
            decoder_out, r_decoder_out, _ = decoder(
                encoder_out,
                encoder_mask,
                hyps,
                ys_in_lens,
                r_hyps,
                reverse_weight
            )
            
            # 合并正向和反向得分
            if reverse_weight > 0:
                # r_decoder_out 也是 [beam, len, vocab]
                # 反向输出的最后一个位置对应正向的第一个预测
                # 但这里我们简化处理，只用正向
                pass
        else:
            # 普通 TransformerDecoder
            decoder_out, _ = decoder(
                encoder_out,
                encoder_mask,
                hyps,
                ys_in_lens
            )
        
        # 取最后一个位置的 logits
        logits = decoder_out[:, -1, :]  # [beam, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
        
        vocab_size = log_probs.size(-1)
        
        # 计算新分数
        if step == prefix_len:
            # 第一步，只从第一个 beam 扩展
            next_scores = scores[0] + log_probs[0]
            topk_scores, topk_tokens = torch.topk(next_scores, beam_size)
            beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
        else:
            next_scores = scores.unsqueeze(1) + log_probs  # [beam, vocab]
            next_scores = next_scores.view(-1)  # [beam * vocab]
            topk_scores, topk_indices = torch.topk(next_scores, beam_size)
            beam_indices = topk_indices // vocab_size
            topk_tokens = topk_indices % vocab_size
        
        # 更新 hyps
        new_hyps = torch.cat([
            hyps[beam_indices],
            topk_tokens.unsqueeze(1)
        ], dim=1)
        
        scores = topk_scores
        hyps = new_hyps
        
        # 检查结束
        ended_mask = (topk_tokens == eos_id)
        for i in range(beam_size):
            if ended_mask[i]:
                ended_hyps.append({
                    'score': scores[i].item(),
                    'tokens': hyps[i].tolist()
                })
        
        if len(ended_hyps) >= beam_size:
            break
    
    # 补充未结束的
    if len(ended_hyps) < beam_size:
        for i in range(beam_size):
            if len(ended_hyps) >= beam_size:
                break
            ended_hyps.append({
                'score': scores[i].item(),
                'tokens': hyps[i].tolist()
            })
    
    # 排序
    ended_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)
    
    # 转换格式
    results = []
    for hyp in ended_hyps[:beam_size]:
        tokens = hyp['tokens']
        if tokens and tokens[0] == sos_id:
            tokens = tokens[1:]
        if tokens and tokens[-1] == eos_id:
            tokens = tokens[:-1]
        results.append(DecodeResult(tokens=tokens, score=hyp['score']))
    
    return results


def attention_decode_with_prefix(model, feats, feats_lengths, beam_size,
                                  prefix_ids, sos_id, eos_id, device,
                                  reverse_weight=0.0,
                                  decoding_chunk_size=-1,
                                  num_decoding_left_chunks=-1,
                                  **kwargs):
    """
    带前缀的 Attention 解码
    """
    # 前缀序列包含 sos
    prefix_with_sos = [sos_id] + list(prefix_ids)
    logging.debug(f"前缀序列 (含sos): {prefix_with_sos}")
    
    # Encoder forward
    encoder_out, encoder_mask = model.encoder(
        feats, 
        feats_lengths,
        decoding_chunk_size=decoding_chunk_size,
        num_decoding_left_chunks=num_decoding_left_chunks
    )
    
    # Beam search with prefix
    hyps = beam_search_with_prefix(
        model, encoder_out, encoder_mask, beam_size,
        prefix_with_sos, sos_id, eos_id, device,
        reverse_weight=reverse_weight
    )
    
    return hyps


def decode_with_prefix(model, modes, feats, feats_lengths, beam_size,
                       prefix_ids, sos_id, eos_id, device,
                       reverse_weight=0.0,
                       **kwargs):
    """
    带前缀的解码主函数
    """
    results = {}
    
    for mode in modes:
        if mode == 'attention' and prefix_ids is not None and len(prefix_ids) > 0:
            hyps = attention_decode_with_prefix(
                model, feats, feats_lengths, beam_size,
                prefix_ids, sos_id, eos_id, device,
                reverse_weight=reverse_weight,
                **kwargs
            )
            results[mode] = hyps
        else:
            # 其他模式用原始解码
            result = model.decode(
                [mode], feats, feats_lengths, beam_size, **kwargs
            )
            results[mode] = result[mode]
    
    return results


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)

    # ============== 加载前缀文件 ==============
    prefix_dict = load_prefix_file(args.prefix_file)
    use_prefix = len(prefix_dict) > 0 and 'attention' in args.modes
    
    if use_prefix:
        logging.info(f"启用前缀引导解码，模式: {args.prefix_mode}")
        if args.prefix_mode == 'n_chars':
            logging.info(f"使用前 {args.prefix_n_chars} 个字符")
    
    # 获取 sos/eos id
    sos_id = tokenizer.symbol_table.get('<sos>', 1)
    eos_id = tokenizer.symbol_table.get('<eos>', 2)
    if hasattr(model, 'sos'):
        sos_id = model.sos
    if hasattr(model, 'eos'):
        eos_id = model.eos
    logging.info(f"sos_id={sos_id}, eos_id={eos_id}")
    # =============================================

    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])

    def run_decode(batch):
        keys = batch["keys"]
        feats = batch["feats"].to(device)
        target = batch["target"].to(device)
        feats_lengths = batch["feats_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        
        infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
        
        if use_prefix and args.batch_size == 1:
            key = keys[0]
            prefix_text = prefix_dict.get(key, '')
            
            if prefix_text:
                prefix_ids = get_prefix_tokens(
                    prefix_text, tokenizer,
                    mode=args.prefix_mode,
                    n_chars=args.prefix_n_chars
                )
                logging.info(f"[{key}] 前缀: '{prefix_text[:args.prefix_n_chars if args.prefix_mode == 'n_chars' else 1]}' -> {prefix_ids}")
                
                results = decode_with_prefix(
                    model, args.modes, feats, feats_lengths,
                    args.beam_size,
                    prefix_ids=prefix_ids,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    device=device,
                    reverse_weight=args.reverse_weight,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    context_graph=context_graph,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    infos=infos
                )
            else:
                results = model.decode(
                    args.modes,
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    context_graph=context_graph,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    infos=infos
                )
        else:
            results = model.decode(
                args.modes,
                feats,
                feats_lengths,
                args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                ctc_weight=args.ctc_weight,
                simulate_streaming=args.simulate_streaming,
                reverse_weight=args.reverse_weight,
                context_graph=context_graph,
                blank_id=blank_id,
                blank_penalty=args.blank_penalty,
                length_penalty=args.length_penalty,
                infos=infos
            )
        
        for i, key in enumerate(keys):
            for mode, hyps in results.items():
                if isinstance(hyps, list) and len(hyps) > i:
                    hyp = hyps[i] if isinstance(hyps, list) else hyps
                    tokens = hyp.tokens if hasattr(hyp, 'tokens') else hyp
                elif hasattr(hyps, 'tokens'):
                    tokens = hyps.tokens
                else:
                    tokens = hyps[0].tokens if hyps else []
                
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                
                line = '{} {}'.format(key, tokenizer.detokenize(tokens)[0])
                logging.info('{} {}'.format(mode.ljust(max_format_len), line))
                files[mode].write(line + '\n')

    if "cuda" in args.device:
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False):
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_data_loader):
                    run_decode(batch)
    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                run_decode(batch)

    for mode, f in files.items():
        f.close()


if __name__ == '__main__':
    main()