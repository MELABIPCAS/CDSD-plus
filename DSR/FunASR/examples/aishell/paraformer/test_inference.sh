#!/bin/bash
# 测试推理功能的小脚本

# 创建测试数据（只用几个utterance）
mkdir -p test_inference

# 复制前5个mild和severe utterance
head -5 inference_by_speaker_20260102_103108/mild_temp.wav.scp > test_inference/mild_test.wav.scp
head -5 inference_by_speaker_20260102_103108/mild_temp.text > test_inference/mild_test.text
head -3 inference_by_speaker_20260102_103108/severe_temp.wav.scp > test_inference/severe_test.wav.scp
head -3 inference_by_speaker_20260102_103108/severe_temp.text > test_inference/severe_test.text

echo "Testing mild speakers with Stage1 model..."
python -m funasr.bin.inference \
    --config-path="exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage1_mild_full" \
    --config-name="config.yaml" \
    ++init_param="/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage1_mild_full/model.pt.avg10" \
    ++tokenizer_conf.token_list="/home/zhaoxr/FunASR/data/cdsd_dependent/data/zh_token_list/char/tokens.txt" \
    ++frontend_conf.cmvn_file="/home/zhaoxr/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn" \
    ++input="test_inference/mild_test.wav.scp" \
    ++output_dir="test_inference/mild_stage1" \
    ++device="cuda:0"

echo "Testing severe speakers with Stage2 model..."
python -m funasr.bin.inference \
    --config-path="exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage2_severe_lora" \
    --config-name="config.yaml" \
    ++init_param="/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage2_severe_lora/model.pt.avg10" \
    ++tokenizer_conf.token_list="/home/zhaoxr/FunASR/data/cdsd_dependent/data/zh_token_list/char/tokens.txt" \
    ++frontend_conf.cmvn_file="/home/zhaoxr/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn" \
    ++input="test_inference/severe_test.wav.scp" \
    ++output_dir="test_inference/severe_stage2" \
    ++device="cuda:0"

echo "Test inference completed!"
echo "Check results in test_inference/ directory"
