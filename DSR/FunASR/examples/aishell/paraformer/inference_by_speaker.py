#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据说话人分类进行推理的脚本
轻度障碍说话人使用Stage1全微调模型
重度障碍说话人使用Stage2 LoRA模型
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def load_speaker_lists():
    """加载说话人分类列表"""
    mild_speakers = set()
    severe_speakers = set()

    # 读取轻度障碍说话人列表
    with open('mild_speakers.txt', 'r', encoding='utf-8') as f:
        for line in f:
            speaker = line.strip()
            if speaker and not speaker.startswith('#'):
                mild_speakers.add(speaker)

    # 读取重度障碍说话人列表
    with open('severe_speakers.txt', 'r', encoding='utf-8') as f:
        for line in f:
            speaker = line.strip()
            if speaker and not speaker.startswith('#') and speaker != '平均CER':
                severe_speakers.add(speaker)

    return mild_speakers, severe_speakers

def get_speaker_from_utterance(utterance_id):
    """从utterance ID中提取说话人ID（前4位）"""
    return utterance_id[:4]

def split_test_data_by_speaker(test_wav_scp, test_text, mild_speakers, severe_speakers):
    """根据说话人分类拆分测试数据"""
    mild_data = []
    severe_data = []
    unknown_data = []

    # 读取text文件，创建utterance到文本的映射
    text_dict = {}
    with open(test_text, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utterance_id = parts[0]
                text = ' '.join(parts[1:])
                text_dict[utterance_id] = text

    # 读取wav.scp文件，根据说话人分类
    with open(test_wav_scp, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utterance_id = parts[0]
                wav_path = parts[1]
                speaker_id = get_speaker_from_utterance(utterance_id)
                text = text_dict.get(utterance_id, '')

                data_entry = {
                    'utterance_id': utterance_id,
                    'wav_path': wav_path,
                    'text': text,
                    'speaker_id': speaker_id
                }

                if speaker_id in mild_speakers:
                    mild_data.append(data_entry)
                elif speaker_id in severe_speakers:
                    severe_data.append(data_entry)
                else:
                    unknown_data.append(data_entry)
                    print(f"Warning: Unknown speaker {speaker_id} for utterance {utterance_id}")

    return mild_data, severe_data, unknown_data

def create_temp_data_files(data_list, prefix):
    """创建临时数据文件"""
    wav_scp_file = f"{prefix}_temp.wav.scp"
    text_file = f"{prefix}_temp.text"

    with open(wav_scp_file, 'w', encoding='utf-8') as f_wav, \
         open(text_file, 'w', encoding='utf-8') as f_text:
        for entry in data_list:
            f_wav.write(f"{entry['utterance_id']} {entry['wav_path']}\n")
            f_text.write(f"{entry['utterance_id']} {entry['text']}\n")

    return wav_scp_file, text_file

def run_inference(model_path, wav_scp_file, text_file, output_dir, gpu_id=0):
    """运行FunASR推理"""
    # 使用正确的FunASR推理命令格式
    config_path = "exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage1_mild_full"  # 使用stage1的config作为基础
    token_list = "/home/zhaoxr/FunASR/data/cdsd_dependent/data/zh_token_list/char/tokens.txt"
    cmvn_file = "/home/zhaoxr/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"

    cmd = [
        'python', '-m', 'funasr.bin.inference',
        f'--config-path={config_path}',
        '--config-name=config.yaml',
        f'++init_param={model_path}',
        f'++tokenizer_conf.token_list={token_list}',
        f'++frontend_conf.cmvn_file={cmvn_file}',
        f'++input={wav_scp_file}',
        f'++output_dir={output_dir}',
        f'++device=cuda:{gpu_id}'
    ]

    print(f"Running inference with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.getcwd()), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running inference: {result.stderr}")
        return False

    print(f"Inference completed successfully. Output saved to {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='根据说话人分类进行推理')
    parser.add_argument('--test_wav_scp', required=True, help='测试集wav.scp文件路径')
    parser.add_argument('--test_text', required=True, help='测试集text文件路径')
    parser.add_argument('--stage1_model', required=True, help='Stage1模型路径')
    parser.add_argument('--stage2_model', required=True, help='Stage2模型路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    # 加载说话人分类
    print("Loading speaker classification lists...")
    mild_speakers, severe_speakers = load_speaker_lists()
    print(f"Mild speakers: {len(mild_speakers)}")
    print(f"Severe speakers: {len(severe_speakers)}")

    # 拆分测试数据
    print("Splitting test data by speaker...")
    mild_data, severe_data, unknown_data = split_test_data_by_speaker(
        args.test_wav_scp, args.test_text, mild_speakers, severe_speakers
    )

    print(f"Mild data: {len(mild_data)} utterances")
    print(f"Severe data: {len(severe_data)} utterances")
    print(f"Unknown data: {len(unknown_data)} utterances")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 对轻度障碍说话人使用Stage1模型进行推理
    if mild_data:
        print("\n=== Running inference for mild speakers using Stage1 model ===")
        mild_wav_scp, mild_text = create_temp_data_files(mild_data, "mild")
        mild_output_dir = os.path.join(args.output_dir, "mild_stage1")

        success = run_inference(args.stage1_model, mild_wav_scp, mild_text, mild_output_dir, args.gpu_id)
        if success:
            print(f"Mild speakers inference completed. Results in {mild_output_dir}")

        # 清理临时文件
        os.remove(mild_wav_scp)
        os.remove(mild_text)

    # 对重度障碍说话人使用Stage2模型进行推理
    if severe_data:
        print("\n=== Running inference for severe speakers using Stage2 model ===")
        severe_wav_scp, severe_text = create_temp_data_files(severe_data, "severe")
        severe_output_dir = os.path.join(args.output_dir, "severe_stage2")

        success = run_inference(args.stage2_model, severe_wav_scp, severe_text, severe_output_dir, args.gpu_id)
        if success:
            print(f"Severe speakers inference completed. Results in {severe_output_dir}")

        # 清理临时文件
        os.remove(severe_wav_scp)
        os.remove(severe_text)

    # 处理未知说话人（如果有的话）
    if unknown_data:
        print(f"\nWarning: {len(unknown_data)} utterances from unknown speakers. Skipping...")

    # 合并结果并计算WER
    print("\n=== Calculating overall WER ===")
    # 这里可以添加WER计算逻辑

    print("Inference completed!")

if __name__ == "__main__":
    main()
