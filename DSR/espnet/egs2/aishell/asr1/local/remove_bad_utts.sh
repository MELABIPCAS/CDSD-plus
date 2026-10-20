#!/usr/bin/env bash

set -euo pipefail

# 针对训练集中损坏的录音做清理，默认移除：
#   S009T013E000N00001 及其 speed perturbation 版本。
# 脚本会在当前 asr1 目录下生成 local/bad_utts.list，
# 并使用 utils/filter_scp.pl 逐个过滤数据目录，再执行 fix_data_dir.sh。

bad_list_path="local/bad_utts.list"
declare -a bad_utts=(
  "S009T013E000N00001"
  "sp0.9-S009T013E000N00001"
  "sp1.1-S009T013E000N00001"
)

declare -a data_dirs=(
  "data/train"
  "data/train_sp"
  "data/train_sp0.9"
  "data/train_sp1.1"
)

declare -a local_files=(
  "data/local/train/wav.scp"
  "data/local/train/text"
  "data/local/train/utt2spk"
  "data/local/train/utt.list"
  "data/local/train/transcripts.txt"
  "data/local/train/wav.scp_all"
  "data/local/train/utt2spk_all"
)

# 生成坏样本列表
mkdir -p "$(dirname "${bad_list_path}")"
: > "${bad_list_path}"
printf "%s\n" "${bad_utts[@]}" >> "${bad_list_path}"

filter_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    return
  fi
  echo "Filtering ${file}"
  utils/filter_scp.pl --exclude "${bad_list_path}" "${file}" > "${file}.tmp"
  mv "${file}.tmp" "${file}"
}

# 过滤数据目录
for dir in "${data_dirs[@]}"; do
  if [[ ! -d "${dir}" ]]; then
    continue
  fi
  for file in wav.scp text utt2spk utt2uniq segments; do
    filter_file "${dir}/${file}"
  done
  if [[ -f "${dir}/spk2utt" ]]; then
    # spk2utt 依赖 utt2spk，后面 fix_data_dir.sh 会重建
    rm -f "${dir}/spk2utt"
  fi
  utils/fix_data_dir.sh "${dir}"
done

# 过滤 local/train 相关文件
for file in "${local_files[@]}"; do
  filter_file "${file}"
done

cat <<'EOF'
坏条目已经从数据描述里移除：
1. 建议清理旧特征，例如：
   rm -rf dump/raw/train* dump/fbank_pitch/train*
2. 然后从生成特征的 stage 重新运行 run.sh。
EOF

