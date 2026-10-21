#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为多任务学习准备数据：统计训练集中的说话人数量
（验证集和测试集的说话人可能不在训练集中，这是正常的）
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def analyze_speakers(train_jsonl, dev_jsonl=None, test_jsonl=None, speaker_id_prefix=4, speaker_id_field=None):
    """
    分析训练集、验证集、测试集中的说话人分布
    
    Returns:
        train_speakers: 训练集中的说话人集合
        dev_speakers: 验证集中的说话人集合（如果有）
        test_speakers: 测试集中的说话人集合（如果有）
        all_speakers: 所有说话人集合
    """
    def extract_speakers(jsonl_path):
        speakers = set()
        speaker_stats = Counter()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if speaker_id_field and speaker_id_field in item:
                        speaker_id = item[speaker_id_field]
                    else:
                        key = item.get("key") or item.get("utt_id")
                        if key is None:
                            continue
                        speaker_id = key[:speaker_id_prefix]
                    speakers.add(speaker_id)
                    speaker_stats[speaker_id] += 1
                except:
                    continue
        return speakers, speaker_stats
    
    train_speakers, train_stats = extract_speakers(train_jsonl)
    all_speakers = train_speakers.copy()
    
    dev_speakers, dev_stats = None, None
    if dev_jsonl and Path(dev_jsonl).exists():
        dev_speakers, dev_stats = extract_speakers(dev_jsonl)
        all_speakers.update(dev_speakers)
    
    test_speakers, test_stats = None, None
    if test_jsonl and Path(test_jsonl).exists():
        test_speakers, test_stats = extract_speakers(test_jsonl)
        all_speakers.update(test_speakers)
    
    return train_speakers, dev_speakers, test_speakers, all_speakers, train_stats, dev_stats, test_stats


def main():
    parser = argparse.ArgumentParser(description="分析多任务学习数据的说话人分布")
    parser.add_argument("train_jsonl", type=str, help="训练集 jsonl 文件")
    parser.add_argument("--dev-jsonl", type=str, default=None, help="验证集 jsonl 文件（可选）")
    parser.add_argument("--test-jsonl", type=str, default=None, help="测试集 jsonl 文件（可选）")
    parser.add_argument("--speaker-id-prefix", type=int, default=4, help="从 key 前 N 个字符提取说话人 ID")
    parser.add_argument("--speaker-id-field", type=str, default=None, help="说话人 ID 字段名")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("多任务学习数据说话人分析")
    print("=" * 70)
    
    train_speakers, dev_speakers, test_speakers, all_speakers, train_stats, dev_stats, test_stats = analyze_speakers(
        args.train_jsonl, args.dev_jsonl, args.test_jsonl,
        args.speaker_id_prefix, args.speaker_id_field
    )
    
    print(f"\n训练集:")
    print(f"  说话人数量: {len(train_speakers)}")
    print(f"  样本总数: {sum(train_stats.values())}")
    print(f"  平均每说话人样本数: {sum(train_stats.values()) / len(train_speakers):.1f}")
    
    if dev_speakers is not None:
        print(f"\n验证集:")
        print(f"  说话人数量: {len(dev_speakers)}")
        print(f"  样本总数: {sum(dev_stats.values())}")
        overlap = len(dev_speakers & train_speakers)
        ood = len(dev_speakers - train_speakers)
        print(f"  与训练集重叠: {overlap} 人")
        print(f"  未见过的说话人: {ood} 人")
    
    if test_speakers is not None:
        print(f"\n测试集:")
        print(f"  说话人数量: {len(test_speakers)}")
        print(f"  样本总数: {sum(test_stats.values())}")
        overlap = len(test_speakers & train_speakers)
        ood = len(test_speakers - train_speakers)
        print(f"  与训练集重叠: {overlap} 人")
        print(f"  未见过的说话人: {ood} 人")
    
    print(f"\n总计:")
    print(f"  所有说话人数量: {len(all_speakers)}")
    
    print("\n" + "=" * 70)
    print("配置建议:")
    print("=" * 70)
    print(f"  # 训练集中只有 {len(train_speakers)} 个说话人，这是正常的")
    print(f"  # 验证集和测试集的说话人不在训练集中，这正是我们要测试的场景")
    print(f"  ++model_conf.num_speakers={len(train_speakers)}  # 只设置训练集的说话人数量")
    print(f"  ++dataset_conf.speaker_id_prefix={args.speaker_id_prefix}")
    if args.speaker_id_field:
        print(f"  ++dataset_conf.speaker_id_field={args.speaker_id_field}")
    print("\n注意：")
    print("  - 训练时，说话人分类任务只学习训练集中的说话人")
    print("  - 验证/测试时，未见过的说话人会被自动忽略（speaker_id=-1）")
    print("  - 多任务学习的目的是学习说话人无关的特征，而不是精确分类说话人")
    print("  - 这样设计可以提升在未见过的说话人上的 ASR 准确率")


if __name__ == "__main__":
    main()


