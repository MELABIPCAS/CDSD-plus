# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# 推理脚本（批量、并行、计算CER），仿照 aishell/paraformer/run.sh 的 stage 5

# 必填：你的本地模型输出目录（内含 config.yaml 与 model.pt 或 model.pt.avgN）
local_path="/home/zhaoxr/FunASR/examples/industrial_data_pretraining/paraformer/outputs"

# 可选：权重文件名（默认使用 model.pt.avg10，若不存在请改为 model.pt）
inference_checkpoint="model.pt"

# 必填：测试清单（wav.scp），以及参考文本（可选，用于计算CER）
input_scp="/home/zhaoxr/FunASR/data/CDSDIDT/data/test/wav.scp"
ref_text="/home/zhaoxr/FunASR/data/CDSDIDT/data/test/text"  # 若没有参考文本，可留空或指向不存在文件

# 输出目录
output_dir="${local_path}/infer_test"

# 设备/GPU 并行相关
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"9"}
device="cuda" # "cuda" 或 "cpu"
inference_batch_size=32

# 可能存在的 tokenizer/cmvn 文件（若存在则自动带上）
tokens="/home/zhaoxr/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json"
cmvn_file="/home/zhaoxr/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"
config="config.yaml"

# 并行作业数与 GPU 列表
if [ "${device}" = "cuda" ]; then
    gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    nj=${gpu_num}
else
    nj=32
    inference_batch_size=1
fi

# 目录与日志
mkdir -p "${output_dir}"
inference_tag=$(basename "${inference_checkpoint}")
inference_dir="${output_dir}/inference-${inference_tag}"
_logdir="${inference_dir}/logdir"
mkdir -p "${_logdir}"

# 分割 scp
key_file="${input_scp}"
# 基本检查
if [ ! -f "${key_file}" ]; then
  echo "Error: input_scp not found: ${key_file}" >&2
  exit 1
fi
split_scps=
for JOB in $(seq "${nj}"); do
    split_scps+=" ${_logdir}/keys.${JOB}.scp"
done
# 使用 split_scp.pl 按行切分 wav.scp
perl /home/zhaoxr/FunASR/examples/aishell/paraformer/utils/split_scp.pl "${key_file}" ${split_scps}

# 组织可选参数
extra_args=()
if [ -f "${tokens}" ]; then
  extra_args+=(++tokenizer_conf.token_list="${tokens}")
fi
if [ -f "${cmvn_file}" ]; then
  extra_args+=(++frontend_conf.cmvn_file="${cmvn_file}")
fi

# 解析 GPU 列表
gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })

# 并行推理
for JOB in $(seq ${nj}); do
  {
    if [ "${device}" = "cuda" ]; then
      id=$((JOB-1))
      gpuid=${gpuid_list_array[$id]}
      export CUDA_VISIBLE_DEVICES=${gpuid}
    else
      export CUDA_VISIBLE_DEVICES=""
    fi

    python -m funasr.bin.inference \
    --config-path "${local_path}" \
    --config-name "${config}" \
    ++init_param="${local_path}/${inference_checkpoint}" \
    ${extra_args[@]:-} \
    ++input="${_logdir}/keys.${JOB}.scp" \
    ++output_dir="${inference_dir}/${JOB}" \
    ++device="${device}" \
    ++ncpu=1 \
    ++disable_log=true \
    ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt
  }&
done
wait

# 合并 1best 输出
mkdir -p ${inference_dir}/1best_recog
for f in token score text; do
  if [ -f "${inference_dir}/1best_recog/${f}" ]; then
    rm -f "${inference_dir}/1best_recog/${f}"
  fi
  if [ -f "${inference_dir}/1best_recog/${f}" ] || [ -f "${inference_dir}/1best_recog/${f}" ]; then
    :
  fi
  have_part=false
  for JOB in $(seq "${nj}"); do
    if [ -f "${inference_dir}/${JOB}/1best_recog/${f}" ]; then
      have_part=true
      break
    fi
  done
  if ${have_part}; then
    for JOB in $(seq "${nj}"); do
      if [ -f "${inference_dir}/${JOB}/1best_recog/${f}" ]; then
        cat "${inference_dir}/${JOB}/1best_recog/${f}"
      fi
    done | sort -k1 >"${inference_dir}/1best_recog/${f}"
  fi
done

# 计算 CER（若提供了参考文本）
if [ -f "${ref_text}" ] && [ -f "${inference_dir}/1best_recog/text" ]; then
  echo "Computing CER ..."
  python ../../aishell/paraformer/utils/postprocess_text_zh.py \
    "${inference_dir}/1best_recog/text" \
    "${inference_dir}/1best_recog/text.proc"

  python ../../aishell/paraformer/utils/postprocess_text_zh.py \
    "${ref_text}" \
    "${inference_dir}/1best_recog/text.ref"

  python ../../aishell/paraformer/utils/compute_wer.py \
    "${inference_dir}/1best_recog/text.ref" \
    "${inference_dir}/1best_recog/text.proc" \
    "${inference_dir}/1best_recog/text.cer"

  tail -n 3 "${inference_dir}/1best_recog/text.cer" || true
else
  echo "Skip CER: ref_text not found or no hypothesis text."
fi




