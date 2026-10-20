# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import math
import logging
import time
from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import (add_sos_eos, log_add, add_whisper_tokens,
                                mask_to_bias)
from wenet.utils.ctc_utils import remove_duplicates_and_blank
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.utils.context_graph import ContextGraph, ContextState


class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


class PrefixScore:
    """ For CTC prefix beam search """

    def __init__(self,
                 s: float = float('-inf'),
                 ns: float = float('-inf'),
                 v_s: float = float('-inf'),
                 v_ns: float = float('-inf'),
                 context_state: ContextState = None,
                 context_score: float = 0.0):
        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score
        self.v_s = v_s  # viterbi blank ending score
        self.v_ns = v_ns  # viterbi none blank ending score
        self.cur_token_prob = float('-inf')  # prob of current token
        self.times_s = []  # times of viterbi blank path
        self.times_ns = []  # times of viterbi none blank path
        self.context_state = context_state
        self.context_score = context_score
        self.has_context = False

    def score(self):
        return log_add(self.s, self.ns)

    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

    def total_score(self):
        return self.score() + self.context_score

    def copy_context(self, prefix_score):
        self.context_score = prefix_score.context_score
        self.context_state = prefix_score.context_state

    def update_context(self, context_graph, prefix_score, word_id):
        self.copy_context(prefix_score)
        (score, context_state) = context_graph.forward_one_step(
            prefix_score.context_state, word_id)
        self.context_score += score
        self.context_state = context_state


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results


def ctc_prefix_beam_search(
    ctc_probs: torch.Tensor,
    ctc_lens: torch.Tensor,
    beam_size: int,
    context_graph: ContextGraph = None,
    blank_id: int = 0,
) -> List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    batch_size = ctc_probs.shape[0]
    results = []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(),
                     PrefixScore(s=0.0,
                                 ns=-float('inf'),
                                 v_s=0.0,
                                 v_ns=0.0,
                                 context_state=None if context_graph is None
                                 else context_graph.root,
                                 context_score=0.0))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for u in top_k_index:
                u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == blank_id:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(next_score.s,
                                               prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        # perfix not changed, copy the context from prefix
                        if context_graph and not next_score.has_context:
                            next_score.copy_context(prefix_score)
                            next_score.has_context = True
                    elif u == last:
                        #  Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns,
                                                 prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy(
                                )
                                next_score1.times_ns[-1] = t
                        if context_graph and not next_score1.has_context:
                            next_score1.copy_context(prefix_score)
                            next_score1.has_context = True

                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u, )
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns,
                                                 prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                        if context_graph and not next_score2.has_context:
                            next_score2.update_context(context_graph,
                                                       prefix_score, u)
                            next_score2.has_context = True
                    else:
                        n_prefix = prefix + (u, )
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(next_score.ns,
                                                prefix_score.score() + prob)
                        if next_score.v_ns < prefix_score.viterbi_score(
                        ) + prob:
                            next_score.v_ns = prefix_score.viterbi_score(
                            ) + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(t)
                        if context_graph and not next_score.has_context:
                            next_score.update_context(context_graph,
                                                      prefix_score, u)
                            next_score.has_context = True

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: x[1].total_score(),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]

        # We should backoff the context score/state when the context is
        # not fully matched at the last time.
        if context_graph is not None:
            for i, hyp in enumerate(cur_hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp[1].context_state)
                cur_hyps[i][1].context_score = context_score
                cur_hyps[i][1].context_state = new_context_state

        nbest = [y[0] for y in cur_hyps]
        nbest_scores = [y[1].total_score() for y in cur_hyps]
        nbest_times = [y[1].times() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]
        results.append(
            DecodeResult(tokens=best,
                         score=best_score,
                         times=best_time,
                         nbest=nbest,
                         nbest_scores=nbest_scores,
                         nbest_times=nbest_times))
    return results


def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    beam_size: int = 10,
    length_penalty: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    if getattr(model, 'special_tokens', None) is not None \
            and "transcribe" in model.special_tokens:
        tasks, langs = infos["tasks"], infos["langs"]
        tasks = [t for t in tasks for _ in range(beam_size)]
        langs = [l for l in langs for _ in range(beam_size)]
        hyps = torch.ones([running_size, 0], dtype=torch.long,
                          device=device)  # (B*N, 0)
        hyps, _ = add_whisper_tokens(model.special_tokens,
                                     hyps,
                                     model.ignore_id,
                                     tasks=tasks,
                                     no_timestamp=True,
                                     langs=langs,
                                     use_prev=False)
    else:
        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(model.sos)  # (B*N, 1)
    prefix_len = hyps.size(1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                          dtype=torch.float)
    scores = scores.to(device).repeat([batch_size
                                       ]).unsqueeze(1).to(device)  # (B*N, 1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    cache = {
        'self_att_cache': {},
        'cross_att_cache': {},
    }
    if model.decoder.use_sdpa:
        encoder_mask = mask_to_bias(encoder_mask, encoder_out.dtype)
    if hasattr(model, 'decode_maxlen'):
        maxlen = model.decode_maxlen
    logging.info("attention_beam_search: Starting decoder loop, maxlen={}, prefix_len={}, beam_size={}, batch_size={}".format(
        maxlen, prefix_len, beam_size, batch_size))
    # 2. Decoder forward step by step
    for i in range(prefix_len, maxlen + 1):
        step_start_time = time.time()
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            logging.info("attention_beam_search: All beams finished at step {}".format(i))
            break
        # Log more frequently for early steps to catch issues
        step_num = i - prefix_len + 1
        if step_num <= 20 or step_num % 10 == 0 or i == maxlen:
            logging.info("attention_beam_search: Processing step {}/{} (finished beams: {}/{})".format(
                step_num, maxlen - prefix_len + 1, end_flag.sum().item(), running_size))
        # 2.1 Forward decoder step
        if step_num <= 20:
            logging.info("attention_beam_search: Calling forward_one_step at step {}...".format(i))
        mask_start = time.time()
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
            running_size, 1, 1).to(device)  # (B*N, i, i)
        if model.decoder.use_sdpa:
            hyps_mask = mask_to_bias(hyps_mask, encoder_out.dtype)
        mask_time = time.time() - mask_start
        # logp: (B*N, vocab)
        forward_start = time.time()
        logp = model.decoder.forward_one_step(encoder_out, encoder_mask, hyps,
                                              hyps_mask, cache)
        forward_time = time.time() - forward_start
        if step_num <= 20:
            # Ensure logp is ready before proceeding
            if logp.is_cuda:
                torch.cuda.synchronize()
            logging.info("attention_beam_search: forward_one_step completed at step {} (mask: {:.3f}s, forward: {:.3f}s)".format(
                i, mask_time, forward_time))
            logging.info("attention_beam_search: Step {} logp received: shape={}, device={}".format(
                i, logp.shape, logp.device))
        elif step_num % 50 == 0:
            if logp.is_cuda:
                torch.cuda.synchronize()
            logging.info("attention_beam_search: forward_one_step completed at step {} (forward: {:.3f}s)".format(
                i, forward_time))
        # 2.2 First beam prune: select topk best prob at current time
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} starting pruning...".format(i))
            logging.info("attention_beam_search: Step {} logp shape: {}, dtype: {}, device: {}".format(
                i, logp.shape, logp.dtype, logp.device))
            logging.info("attention_beam_search: Step {} logp has NaN: {}, has Inf: {}".format(
                i, torch.isnan(logp).any().item(), torch.isinf(logp).any().item()))
            if logp.is_cuda:
                torch.cuda.synchronize()
                logging.info("attention_beam_search: Step {} CUDA synchronized before topk".format(i))
        prune_start = time.time()
        topk_start = time.time()
        if step_num <= 20:
            if logp.is_cuda:
                # Check GPU memory before topk
                torch.cuda.synchronize()  # Ensure all previous operations are done
                allocated = torch.cuda.memory_allocated(logp.device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(logp.device) / 1024**3  # GB
                logging.info("attention_beam_search: Step {} GPU memory before topk - allocated: {:.2f}GB, reserved: {:.2f}GB".format(
                    i, allocated, reserved))
                # Force flush CUDA streams
                torch.cuda.synchronize()
            logging.info("attention_beam_search: Step {} calling logp.topk(beam_size={})...".format(i, beam_size))
        try:
            # Ensure logp is contiguous and detached if needed
            if not logp.is_contiguous():
                logp = logp.contiguous()
            
            # Force detach and clone to ensure clean tensor state
            if logp.is_cuda and step_num <= 3:
                # For first few steps, try CPU topk as fallback if GPU hangs
                logp_cpu = logp.detach().cpu()
                top_k_logp_cpu, top_k_index_cpu = logp_cpu.topk(beam_size, dim=-1)
                top_k_logp = top_k_logp_cpu.to(logp.device)
                top_k_index = top_k_index_cpu.to(logp.device)
                logging.info("attention_beam_search: Step {} used CPU topk fallback".format(i))
            else:
                # Normal GPU topk
                top_k_logp, top_k_index = logp.topk(beam_size, dim=-1)  # (B*N, N), explicitly specify dim
            
            # Immediately synchronize and log after topk to confirm it completed
            if logp.is_cuda:
                torch.cuda.synchronize()
            if step_num <= 20:
                logging.info("attention_beam_search: Step {} topk RETURNED successfully!".format(i))
        except Exception as e:
            logging.error("attention_beam_search: Step {} topk failed with error: {}".format(i, str(e)))
            raise
        if step_num <= 20:
            # Only compute stats after topk completes to avoid blocking
            if not logp.is_cuda or step_num <= 5:
                logging.info("attention_beam_search: Step {} logp stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
                    i, logp.min().item(), logp.max().item(), logp.mean().item()))
            if logp.is_cuda:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(logp.device) / 1024**3  # GB
                logging.info("attention_beam_search: Step {} GPU memory after topk - allocated: {:.2f}GB".format(
                    i, allocated))
            logging.info("attention_beam_search: Step {} topk done ({:.3f}s)".format(i, time.time() - topk_start))
            logging.info("attention_beam_search: Step {} topk results - logp shape: {}, index shape: {}".format(
                i, top_k_logp.shape, top_k_index.shape))
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} before mask_finished_scores...".format(i))
        mask_start = time.time()
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} mask_finished_scores done, calling mask_finished_preds...".format(i))
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} mask done ({:.3f}s)".format(i, time.time() - mask_start))
        # 2.3 Second beam prune: select topk score with history
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} starting second beam prune, scores shape: {}, top_k_logp shape: {}".format(
                i, scores.shape, top_k_logp.shape))
        scores_start = time.time()
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} calling scores + top_k_logp (broadcast add)...".format(i))
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} broadcast add done, calling view...".format(i))
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} view done, calling second topk with k={}...".format(i, beam_size))
            if logp.is_cuda:
                torch.cuda.synchronize()
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} second topk done ({:.3f}s)".format(i, time.time() - scores_start))
        prune_time = time.time() - prune_start
        # Update cache to be consistent with new topk scores / hyps
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} starting cache update...".format(i))
        cache_start = time.time()
        index_start = time.time()
        cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
        base_cache_index = (torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
        cache_index = base_cache_index + cache_index
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} cache_index computed ({:.3f}s)".format(i, time.time() - index_start))
        select_start = time.time()
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} cache has {} layers, starting torch.index_select...".format(
                i, len(cache['self_att_cache'])))
        cache['self_att_cache'] = {
            i_layer: (torch.index_select(value[0], dim=0, index=cache_index),
                      torch.index_select(value[1], dim=0, index=cache_index))
            for (i_layer, value) in cache['self_att_cache'].items()
        }
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} torch.index_select done ({:.3f}s)".format(i, time.time() - select_start))
        # NOTE(Mddct): we don't need select cross att here
        empty_start = time.time()
        torch.cuda.empty_cache()
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} torch.cuda.empty_cache done ({:.3f}s)".format(i, time.time() - empty_start))
        cache_time = time.time() - cache_start
        scores = scores.view(-1, 1)  # (B*N, 1)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} pruning/cache done (prune: {:.3f}s, cache: {:.3f}s, total_step: {:.3f}s)".format(
                i, prune_time, cache_time, time.time() - step_start_time))
        # 2.4. Compute base index in top_k_index,
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
        # then find offset_k_index in top_k_index
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} starting step 2.4 (compute base index)...".format(i))
        base_k_start = time.time()
        base_k_index = torch.arange(batch_size, device=device).view(
            -1, 1).repeat([1, beam_size])  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} step 2.4 done ({:.3f}s)".format(i, time.time() - base_k_start))

        # 2.5 Update best hyps
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} starting step 2.5 (update best hyps)...".format(i))
        hyps_start = time.time()
        best_k_pred = torch.index_select(top_k_index.view(-1),
                                         dim=-1,
                                         index=best_k_index)  # (B*N)
        best_hyps_index = best_k_index // beam_size
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} best_k_pred done, starting last_best_k_hyps...".format(i))
        last_best_k_hyps = torch.index_select(
            hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        if step_num <= 20:
            logging.info("attention_beam_search: Step {} step 2.5 done ({:.3f}s)".format(i, time.time() - hyps_start))
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                         dim=1)  # (B*N, i+1)

        # 2.6 Update end flag
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)

    # 3. Select best of best
    logging.info("attention_beam_search: Decoder loop completed, selecting best hypotheses...")
    scores = scores.view(batch_size, beam_size)
    lengths = hyps.ne(model.eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(
        batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, prefix_len:]
    logging.info("attention_beam_search: Best hypotheses selected, preparing results...")

    results = []
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        results.append(DecodeResult(hyp.tolist()))
    return results


def attention_rescoring(
    model,
    ctc_prefix_results: List[DecodeResult],
    encoder_outs: torch.Tensor,
    encoder_lens: torch.Tensor,
    ctc_weight: float = 0.0,
    reverse_weight: float = 0.0,
    infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    sos, eos = model.sos_symbol(), model.eos_symbol()
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = ctc_prefix_results[b].nbest
        ctc_scores = ctc_prefix_results[b].nbest_scores
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, model.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        if getattr(model, 'special_tokens', None) is not None \
                and "transcribe" in model.special_tokens:
            prev_len = hyps_pad.size(1)
            hyps_pad, _ = add_whisper_tokens(
                model.special_tokens,
                hyps_pad,
                model.ignore_id,
                tasks=[infos["tasks"][b]] * len(hyps),
                no_timestamp=True,
                langs=[infos["langs"][b]] * len(hyps),
                use_prev=False)
            cur_len = hyps_pad.size(1)
            hyps_lens = hyps_lens + cur_len - prev_len
            prefix_len = 4
        else:
            hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
            hyps_lens = hyps_lens + 1  # Add <sos> at begining
            prefix_len = 1
        decoder_out, r_decoder_out = model.forward_attention_decoder(
            hyps_pad, hyps_lens, encoder_out, reverse_weight)
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        confidences = []
        tokens_confidences = []
        for i, hyp in enumerate(hyps):
            score = 0.0
            tc = []  # tokens confidences
            for j, w in enumerate(hyp):
                s = decoder_out[i][j + (prefix_len - 1)][w]
                score += s
                tc.append(math.exp(s))
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            # add right to left decoder score
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    s = r_decoder_out[i][len(hyp) - j - 1 +
                                         (prefix_len - 1)][w]
                    r_score += s
                    tc[j] = (tc[j] + math.exp(s)) / 2
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            confidences.append(math.exp(score / (len(hyp) + 1)))
            # add ctc score
            score += ctc_scores[i] * ctc_weight
            if score > best_score:
                best_score = score.item()
                best_index = i
            tokens_confidences.append(tc)
        results.append(
            DecodeResult(hyps[best_index],
                         best_score,
                         confidence=confidences[best_index],
                         times=ctc_prefix_results[b].nbest_times[best_index],
                         tokens_confidence=tokens_confidences[best_index]))
    return results
