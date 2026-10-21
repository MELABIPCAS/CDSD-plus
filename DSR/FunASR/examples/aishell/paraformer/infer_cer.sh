#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# 核心：用 token_list.txt 定义的词表做统一分词（匹配模型 token 输出格式）
workspace=$(realpath `pwd`)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}
inference_device="cuda"
inference_batch_size=16
inference_checkpoint="model.pt.ep20"

# 测试集配置
test_data_dir="/home/zhaoxr/snap/data/cdsdzishiyingtest/test"
wav_scp="${test_data_dir}/wav.scp"  # 格式：S077M001W001 /绝对路径.wav
transcript_txt="${test_data_dir}/transcript.txt"
output_root="${workspace}/exp_finetune_5min_concat_lr0.0002/paraformer_fast_adapt/inference_zishiying"
output_dir="${output_root}"  # 主输出目录
log_dir="${output_root}/log"  # 日志目录

# 模型相关配置（关键：token_list.txt 是分词词典）
model_dir="${workspace}/exp_finetune_5min_concat_lr0.0002/paraformer_fast_adapt"
pretrained_model_dir="/home/zhaoxr/FunASR/examples/aishell/paraformer/zishiying_pretrain"
token_list="${pretrained_model_dir}/tokens.txt"  # 分词词典
cmvn_file="${pretrained_model_dir}/am.mvn"
config_file="${model_dir}/config.yaml"

# 并行任务数（单GPU）
nj=1

echo "===== 启动最终测试阶段 ====="
echo "GPU: ${CUDA_VISIBLE_DEVICES} (共${nj}块)"
echo "wav.scp: ${wav_scp}（$(wc -l < ${wav_scp})条）"
echo "模型: ${model_dir}/${inference_checkpoint}"
echo "分词词典: ${token_list}"
echo "输出目录: ${output_dir}"
echo "日志目录: ${log_dir}"
echo "核心逻辑：用 token_list.txt 做统一分词（匹配模型 token 格式）"
echo "==========================="

# 前置检查函数（避免文件缺失/为空）
check_exists() {
    [ ! -f "$1" ] && { echo "错误：文件不存在 $1"; exit 1; }
    [ $(wc -l < "$1") -eq 0 ] && { echo "错误：文件为空 $1"; exit 1; }
}

# 检查核心文件（重点检查 token_list.txt）
check_exists "${wav_scp}"
check_exists "${transcript_txt}"
check_exists "${model_dir}/${inference_checkpoint}"
check_exists "${config_file}"
check_exists "${token_list}"  # 确保分词词典存在
check_exists "${cmvn_file}"

# 提前创建输出/日志目录（确保文件能正常写入）
mkdir -p "${output_dir}" "${log_dir}"
echo "已创建输出目录：${output_dir}"
echo "已创建日志目录：${log_dir}"

# 步骤1：修正wav.scp中的路径错误（适配实际文件存储）
fixed_wav_scp="${log_dir}/wav_scp_final.scp"
cp "${wav_scp}" "${fixed_wav_scp}"
# 修正路径前缀（如需根据实际情况修改）
sed -i "s#S00/#S005/#g" "${fixed_wav_scp}"  # 修正S00→S005路径
sed -i "s#S03/#S034/#g" "${fixed_wav_scp}"  # 修正S03→S034路径
echo "修正后的wav.scp示例（前3条）："
head -n 3 "${fixed_wav_scp}"

# 步骤2：单GPU推理（生成 text 和 token 文件，token 是模型分词结果）
echo "启动推理（单GPU）..."
python "${workspace}/../../../funasr/bin/inference.py" \
--config-path "${model_dir}" \
--config-name "config.yaml" \
++init_param="${model_dir}/${inference_checkpoint}" \
++tokenizer_conf.token_list="${token_list}" \
++frontend_conf.cmvn_file="${cmvn_file}" \
++input="${fixed_wav_scp}" \
++input_format="scp" \
++output_dir="${output_dir}" \
++output_name="1best_recog" \
++device="${inference_device}" \
++batch_size="${inference_batch_size}" \
++disable_log=false \
&> "${log_dir}/inference.log"  # 日志重定向，避免刷屏

# 检查推理结果（text: 完整文本，token: 模型分词结果）
infer_text="${output_dir}/1best_recog/text"
infer_token="${output_dir}/1best_recog/token"
if [ ! -f "${infer_text}" ] || [ $(wc -l < "${infer_text}") -eq 0 ]; then
    echo "错误：推理未生成text结果！查看日志：${log_dir}/inference.log"
    exit 1
fi
if [ ! -f "${infer_token}" ] || [ $(wc -l < "${infer_token}") -eq 0 ]; then
    echo "错误：推理未生成token文件！查看日志：${log_dir}/inference.log"
    exit 1
fi
echo "推理完成！"
echo "生成识别结果（完整文本）：${infer_text}（$(wc -l < ${infer_text})条）"
echo "生成模型分词结果：${infer_token}（$(wc -l < ${infer_token})条）"

# 步骤3：验证模型分词格式（确认 token 文件的分词方式）
echo "======================================"
echo "模型 token 文件格式（前5条，作为分词基准）："
head -n 5 "${infer_token}"
echo "======================================"

# 步骤4：用 token_list.txt 做统一分词（核心步骤）
# 4.1 生成分词脚本（用 Python 实现基于 token_list 的精确分词，避免shell局限性）
tokenize_script="${log_dir}/tokenize_with_list.py"
cat > "${tokenize_script}" << 'EOF'
import sys
from pathlib import Path

def load_token_list(token_list_path):
    """加载 token_list.txt，返回分词词典（按长度降序排序，优先匹配长token）"""
    tokens = []
    with open(token_list_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            token = line.strip()
            if token and token not in ['<blank>', '<unk>', '<s>', '</s>']:  # 跳过特殊符号
                tokens.append(token)
    # 按 token 长度降序排序（避免短token截断长token，如"北京市"优先于"北京"）
    tokens.sort(key=lambda x: len(x), reverse=True)
    return set(tokens), tokens

def tokenize_text(text, token_set, token_sorted):
    """基于分词词典的精确分词（贪婪匹配）"""
    tokenized = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        # 优先尝试最长的token匹配
        for token in token_sorted:
            token_len = len(token)
            if i + token_len <= n and text[i:i+token_len] == token:
                tokenized.append(token)
                i += token_len
                matched = True
                break
        if not matched:
            # 无匹配token时，按单个字符拆分（兼容未登录词）
            tokenized.append(text[i])
            i += 1
    return ' '.join(tokenized)

def process_file(input_path, output_path, token_set, token_sorted):
    """处理输入文件（格式：utt-id 文本），输出分词后的文件"""
    with open(input_path, 'r', encoding='utf-8-sig') as f_in, \
         open(output_path, 'w', encoding='utf-8-sig') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                f_out.write(f"{parts[0]}\n")
                continue
            utt_id, text = parts[0], parts[1]
            # 分词
            tokenized_text = tokenize_text(text, token_set, token_sorted)
            # 写入输出
            f_out.write(f"{utt_id} {tokenized_text}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法：python tokenize_with_list.py <token_list_path> <input_file> <output_file>")
        sys.exit(1)
    token_list_path = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # 加载分词词典
    token_set, token_sorted = load_token_list(token_list_path)
    print(f"加载分词词典：共 {len(token_set)} 个有效token")
    
    # 处理文件
    process_file(input_file, output_file, token_set, token_sorted)
    print(f"分词完成：{input_file} → {output_file}")
EOF

# 4.2 用 token_list.txt 对参考文本和识别结果做统一分词
ref_proc="${output_dir}/1best_recog/text.ref"  # 分词后的参考文本
hyp_proc="${output_dir}/1best_recog/text.proc"  # 分词后的识别结果

echo "用 token_list.txt 分词（匹配模型逻辑）..."
# 处理参考文本（transcript.txt）
python "${tokenize_script}" "${token_list}" "${transcript_txt}" "${ref_proc}"
# 处理识别结果（infer_text，完整文本）
python "${tokenize_script}" "${token_list}" "${infer_text}" "${hyp_proc}"

# 4.3 验证分词结果（和模型 token 文件对比格式）
echo "======================================"
echo "分词后参考文本示例（前2条）："
head -n 2 "${ref_proc}"
echo "分词后识别结果示例（前2条）："
head -n 2 "${hyp_proc}"
echo "模型 token 文件示例（前2条，用于对比）："
head -n 2 "${infer_token}"
echo "======================================"

# 步骤5：修复编码问题，计算CER
cer_result="${output_dir}/1best_recog/text.cer"  # CER报告
wer_tool="${workspace}/../../../funasr/utils/compute_wer.py"
[ ! -f "${wer_tool}" ] && wer_tool="${workspace}/utils/compute_wer.py"
check_exists "${wer_tool}"

# 检查分词后的文件
check_exists "${ref_proc}"
check_exists "${hyp_proc}"

# 修复 compute_wer.py 的编码读取（避免 UnicodeDecodeError）
echo "修复 compute_wer.py 编码读取逻辑..."
sed -i.bak "s/with open(ref_file, 'r')/with open(ref_file, 'r', encoding='utf-8-sig')/g" "${wer_tool}"
sed -i "s/with open(hyp_file, 'r')/with open(hyp_file, 'r', encoding='utf-8-sig')/g" "${wer_tool}"

echo "计算CER（字错率）..."
python "${wer_tool}" "${ref_proc}" "${hyp_proc}" "${cer_result}"

# 恢复原 compute_wer.py（避免修改原始文件）
mv -f "${wer_tool}.bak" "${wer_tool}" 2>/dev/null || true

# 步骤6：输出最终结果（对比分词一致性）
echo "======================================"
echo "测试全流程完成！🎉"
echo "--------------------------------------"
echo "核心文件路径："
echo "  - 原始识别结果（完整文本）：${infer_text}"
echo "  - 模型分词结果（基准）：${infer_token}"
echo "  - 分词后参考文本（token_list 分词）：${ref_proc}"
echo "  - 分词后识别结果（token_list 分词）：${hyp_proc}"
echo "  - CER报告：${cer_result}"
echo "--------------------------------------"
echo "分词一致性验证："
echo "模型 token 文件行数：$(wc -l < ${infer_token})"
echo "分词后识别结果行数：$(wc -l < ${hyp_proc})"
echo "--------------------------------------"
echo "CER准确率汇总："
grep -E "%WER|%SER|Scored" "${cer_result}"  # 显示关键指标
echo "======================================"