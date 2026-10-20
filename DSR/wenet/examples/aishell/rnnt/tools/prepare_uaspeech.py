#!/usr/bin/env python3

import os
import json
import random
import argparse
from pathlib import Path

def prepare_uaspeech_processed_data(
    normal_wav_dir,      # 健康对照组音频目录
    dysarthria_wav_dir,  # 构音障碍组音频目录
    normal_text_file,    # 健康对照组文本文件
    dysarthria_text_file, # 构音障碍组文本文件
    output_dir,
    train_ratio=0.9      # 训练集比例，剩余10%作为验证集
):
    """准备已处理的UA-Speech数据集（只有训练集和验证集）"""
    
    # 读取文本文件
    def read_text_file(text_file):
        texts = {}
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    wav_name, text = parts
                    texts[wav_name] = text
        return texts
    
    normal_texts = read_text_file(normal_text_file)
    dysarthria_texts = read_text_file(dysarthria_text_file)
    
    print(f"健康对照组文本: {len(normal_texts)}条")
    print(f"构音障碍组文本: {len(dysarthria_texts)}条")
    
    # 收集所有样本
    all_samples = []
    
    # 处理健康对照组
    print("处理健康对照组...")
    for speaker_dir in Path(normal_wav_dir).iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            print(f"  处理说话人: {speaker_id}")
            
            for wav_file in speaker_dir.glob("*.wav"):
                wav_name = wav_file.stem  # 去掉.wav后缀
                full_utt_id = f"{speaker_id}_{wav_name}"
                
                if wav_name in normal_texts:
                    all_samples.append({
                        'key': full_utt_id,
                        'wav': str(wav_file.absolute()),
                        'txt': normal_texts[wav_name],
                        'type': 'normal',
                        'speaker': speaker_id,
                        'dataset': 'uaspeech'
                    })
                else:
                    print(f"    警告: 未找到文本 {wav_name}")
    
    # 处理构音障碍组
    print("处理构音障碍组...")
    for speaker_dir in Path(dysarthria_wav_dir).iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            print(f"  处理说话人: {speaker_id}")
            
            for wav_file in speaker_dir.glob("*.wav"):
                wav_name = wav_file.stem  # 去掉.wav后缀
                full_utt_id = f"{speaker_id}_{wav_name}"
                
                if wav_name in dysarthria_texts:
                    all_samples.append({
                        'key': full_utt_id,
                        'wav': str(wav_file.absolute()),
                        'txt': dysarthria_texts[wav_name],
                        'type': 'dysarthria',
                        'speaker': speaker_id,
                        'dataset': 'uaspeech'
                    })
                else:
                    print(f"    警告: 未找到文本 {wav_name}")
    
    print(f"总样本数: {len(all_samples)}")
    print(f"健康对照组: {len([s for s in all_samples if s['type'] == 'normal'])}")
    print(f"构音障碍组: {len([s for s in all_samples if s['type'] == 'dysarthria'])}")
    
    # 按说话人分组
    speaker_samples = {}
    for sample in all_samples:
        speaker = sample['speaker']
        if speaker not in speaker_samples:
            speaker_samples[speaker] = []
        speaker_samples[speaker].append(sample)
    
    print(f"总说话人数: {len(speaker_samples)}")
    for speaker, samples in speaker_samples.items():
        normal_count = len([s for s in samples if s['type'] == 'normal'])
        dysarthria_count = len([s for s in samples if s['type'] == 'dysarthria'])
        print(f"  {speaker}: {len(samples)}个样本 (正常:{normal_count}, 构音障碍:{dysarthria_count})")
    
    # 随机分配说话人到训练集和验证集
    speakers = list(speaker_samples.keys())
    random.shuffle(speakers)
    
    n_speakers = len(speakers)
    n_train = int(n_speakers * train_ratio)
    
    train_speakers = speakers[:n_train]
    dev_speakers = speakers[n_train:]
    
    print(f"训练集说话人: {len(train_speakers)} - {train_speakers}")
    print(f"验证集说话人: {len(dev_speakers)} - {dev_speakers}")
    
    # 分配样本到不同集合
    def write_data_list(speakers, split_name):
        samples = []
        for speaker in speakers:
            samples.extend(speaker_samples[speaker])
        
        random.shuffle(samples)
        
        output_file = os.path.join(output_dir, f"{split_name}/data.list")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        normal_count = len([s for s in samples if s['type'] == 'normal'])
        dysarthria_count = len([s for s in samples if s['type'] == 'dysarthria'])
        print(f"{split_name}集: {len(samples)}个样本 (正常:{normal_count}, 构音障碍:{dysarthria_count})")
        return samples
    
    train_samples = write_data_list(train_speakers, "train")
    dev_samples = write_data_list(dev_speakers, "dev")
    
    # 创建wav.scp和text文件
    for split in ["train", "dev"]:
        data_list = os.path.join(output_dir, f"{split}/data.list")
        split_dir = os.path.join(output_dir, split)
        
        wav_scp = []
        text = []
        
        with open(data_list, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                wav_scp.append(f"{sample['key']} {sample['wav']}")
                text.append(f"{sample['key']} {sample['txt']}")
        
        with open(os.path.join(split_dir, 'wav.scp'), 'w', encoding='utf-8') as f:
            for line in wav_scp:
                f.write(line + '\n')
        
        with open(os.path.join(split_dir, 'text'), 'w', encoding='utf-8') as f:
            for line in text:
                f.write(line + '\n')
    
    print("UA-Speech数据准备完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备已处理的UA-Speech数据集（只有训练集和验证集）')
    parser.add_argument('--normal_wav_dir', default="/home/zhaoxr/data/uaspeech/noisereduce_C/UASpeech/audio/healthy_M5", help='健康对照组音频目录')
    parser.add_argument('--dysarthria_wav_dir', default="/home/zhaoxr/data/uaspeech/noisereduce_FM/UASpeech/audio/dysarthria_M5", help='构音障碍组音频目录')
    parser.add_argument('--normal_text_file', default="/home/zhaoxr/data/uaspeech/noisereduce_C/UASpeech/healthy_text.txt", help='健康对照组文本文件')
    parser.add_argument('--dysarthria_text_file', default="/home/zhaoxr/data/uaspeech/noisereduce_FM/UASpeech/dysartjria_text.txt", help='构音障碍组文本文件')
    parser.add_argument('--output_dir', default="/home/zhaoxr/data/uaspeech/data", help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='训练集比例（剩余作为验证集）')
    
    args = parser.parse_args()
    prepare_uaspeech_processed_data(
        args.normal_wav_dir,
        args.dysarthria_wav_dir,
        args.normal_text_file,
        args.dysarthria_text_file,
        args.output_dir,
        args.train_ratio
    )
