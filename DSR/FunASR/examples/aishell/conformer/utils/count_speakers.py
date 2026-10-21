#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 jsonl 文件中的说话人数量
用于设置多任务学习模型中的 num_speakers 参数
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def count_speakers(jsonl_path, speaker_id_prefix=4, speaker_id_field=None):
    """
    统计 jsonl 文件中的说话人数量
    
    Args:
        jsonl_path: jsonl 文件路径
        speaker_id_prefix: 从 key 前 N 个字符提取说话人 ID（默认 4）
        speaker_id_field: jsonl 中的说话人 ID 字段名（如果提供则优先使用）
    
    Returns:
        speaker_count: 说话人数量
        speaker_list: 说话人列表
        speaker_stats: 每个说话人的样本数统计
    """
    speaker_ids = []
    speaker_stats = Counter()
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # 提取说话人 ID
                if speaker_id_field and speaker_id_field in item:
                    speaker_id = item[speaker_id_field]
                else:
                    key = item.get("key") or item.get("utt_id")
                    if key is None:
                        print(f"WARNING: 第 {line_num} 行缺少 key，跳过")
                        continue
                    speaker_id = key[:speaker_id_prefix]
                
                speaker_ids.append(speaker_id)
                speaker_stats[speaker_id] += 1
                
            except json.JSONDecodeError as e:
                print(f"WARNING: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    unique_speakers = sorted(set(speaker_ids))
    return len(unique_speakers), unique_speakers, speaker_stats


def main():
    parser = argparse.ArgumentParser(description="统计 jsonl 文件中的说话人数量")
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="jsonl 文件路径",
    )
    parser.add_argument(
        "--speaker-id-prefix",
        type=int,
        default=4,
        help="从 key 前 N 个字符提取说话人 ID（默认 4）",
    )
    parser.add_argument(
        "--speaker-id-field",
        type=str,
        default=None,
        help="jsonl 中的说话人 ID 字段名（如果提供则优先使用）",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="显示每个说话人的样本数统计",
    )
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"错误: 文件不存在: {jsonl_path}")
        return
    
    print(f"正在统计: {jsonl_path}")
    print(f"说话人 ID 提取方式: ", end="")
    if args.speaker_id_field:
        print(f"字段 '{args.speaker_id_field}'")
    else:
        print(f"key 前 {args.speaker_id_prefix} 个字符")
    print("-" * 60)
    
    speaker_count, speaker_list, speaker_stats = count_speakers(
        jsonl_path,
        speaker_id_prefix=args.speaker_id_prefix,
        speaker_id_field=args.speaker_id_field,
    )
    
    print(f"说话人总数: {speaker_count}")
    print(f"总样本数: {sum(speaker_stats.values())}")
    print(f"平均每个说话人的样本数: {sum(speaker_stats.values()) / speaker_count:.1f}")
    
    if args.show_stats:
        print("\n每个说话人的样本数:")
        for speaker_id in speaker_list:
            count = speaker_stats[speaker_id]
            print(f"  {speaker_id}: {count} 条")
    
    print("\n" + "=" * 60)
    print("配置建议:")
    print(f"  ++model_conf.num_speakers={speaker_count}")
    print(f"  ++dataset_conf.speaker_id_prefix={args.speaker_id_prefix}")
    if args.speaker_id_field:
        print(f"  ++dataset_conf.speaker_id_field={args.speaker_id_field}")


if __name__ == "__main__":
    main()

