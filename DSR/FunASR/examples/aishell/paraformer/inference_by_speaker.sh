#!/bin/bash
# 根据说话人分类进行推理的脚本
# 轻度障碍说话人使用Stage1全微调模型
# 重度障碍说话人使用Stage2 LoRA模型

set -e

# 参数设置
test_data_dir="/home/zhaoxr/FunASR/data/cdsd_dependent/data/test"
stage1_model="/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage1_mild_full/model.pt.avg10"
stage2_model="/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage2_severe_full/model.pt.avg10"
output_dir="/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage2_severe_full/inference_by_speaker_g1g2"

# Multi-job inference settings (similar to run.sh)
CUDA_VISIBLE_DEVICES=${1:-"0,1,2"}  # GPUs to use (first argument)
inference_device="cuda"
inference_batch_size=32

echo "=== Speaker-based Inference ==="
echo "Test data: $test_data_dir"
echo "Stage1 model: $stage1_model"
echo "Stage2 model: $stage2_model"
echo "Output dir: $output_dir"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Batch size: $inference_batch_size"

# Set number of jobs based on available GPUs
nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "Number of parallel jobs: $nj"

# 创建输出目录
mkdir -p "$output_dir"

# 加载说话人分类
mild_speakers=$(cat mild_speakers.txt | grep -v '^#' | grep -v '^$')
severe_speakers=$(cat severe_speakers.txt | grep -v '^#' | grep -v '^$' | grep -v '平均CER')

echo "Mild speakers: $(echo "$mild_speakers" | wc -l)"
echo "Severe speakers: $(echo "$severe_speakers" | wc -l)"

# 创建临时文件来存储按说话人分类的数据
mild_wav_scp="$output_dir/mild_temp.wav.scp"
mild_text="$output_dir/mild_temp.text"
severe_wav_scp="$output_dir/severe_temp.wav.scp"
severe_text="$output_dir/severe_temp.text"

# 清空临时文件
> "$mild_wav_scp"
> "$mild_text"
> "$severe_wav_scp"
> "$severe_text"

# 根据说话人分类处理数据
echo "Classifying utterances by speaker..."
total_count=0
mild_count=0
severe_count=0
unknown_count=0

while IFS= read -r line; do
    utterance_id=$(echo "$line" | awk '{print $1}')
    speaker_id=${utterance_id:0:4}

    # 检查说话人类型
    if echo "$mild_speakers" | grep -q "^${speaker_id}$"; then
        echo "$line" >> "$mild_wav_scp"
        # 找到对应的text行
        text_line=$(grep "^${utterance_id} " "$test_data_dir/text")
        if [ -n "$text_line" ]; then
            echo "$text_line" >> "$mild_text"
        fi
        mild_count=$((mild_count + 1))
    elif echo "$severe_speakers" | grep -q "^${speaker_id}$"; then
        echo "$line" >> "$severe_wav_scp"
        # 找到对应的text行
        text_line=$(grep "^${utterance_id} " "$test_data_dir/text")
        if [ -n "$text_line" ]; then
            echo "$text_line" >> "$severe_text"
        fi
        severe_count=$((severe_count + 1))
    else
        unknown_count=$((unknown_count + 1))
        echo "Warning: Unknown speaker $speaker_id for utterance $utterance_id"
    fi

    total_count=$((total_count + 1))
done < "$test_data_dir/wav.scp"

echo "Classification complete:"
echo "  Total utterances: $total_count"
echo "  Mild speakers: $mild_count"
echo "  Severe speakers: $severe_count"
echo "  Unknown speakers: $unknown_count"

# Function for parallel inference (similar to run.sh)
run_parallel_inference() {
    local input_scp="$1"
    local output_dir="$2"
    local config_path="$3"
    local init_param="$4"
    local speaker_type="$5"

    echo "Running parallel inference for $speaker_type speakers..."
    echo "Input: $input_scp"
    echo "Output: $output_dir"
    echo "Config: $config_path"
    echo "Model: $init_param"

    local _logdir="${output_dir}/logdir"
    mkdir -p "${_logdir}"

    # Split input data into nj parts
    local split_scps=
    for JOB in $(seq "${nj}"); do
        split_scps+=" ${_logdir}/keys.${JOB}.scp"
    done
    utils/split_scp.pl "${input_scp}" ${split_scps}

    # Run parallel jobs
    gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
    for JOB in $(seq ${nj}); do
        {
          id=$((JOB-1))
          gpuid=${gpuid_list_array[$id]}

          echo "Starting job $JOB on GPU $gpuid..."

          python ../../../funasr/bin/inference.py \
            --config-path="${config_path}" \
            --config-name="config.yaml" \
            ++init_param="${init_param}" \
            ++tokenizer_conf.token_list="/home/zhaoxr/FunASR/data/cdsd_dependent/data/zh_token_list/char/tokens.txt" \
            ++frontend_conf.cmvn_file="/home/zhaoxr/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn" \
            ++input="${_logdir}/keys.${JOB}.scp" \
            ++output_dir="${output_dir}/${JOB}" \
            ++device="cuda" \
            ++ncpu=1 \
            ++disable_log=true \
            ++batch_size="${inference_batch_size}" &> ${_logdir}/log.${JOB}.txt
        } &
    done
    wait

    # Merge results
    mkdir -p ${output_dir}/1best_recog
    for f in token score text; do
        if [ -f "${output_dir}/${nj}/1best_recog/${f}" ]; then
            for JOB in $(seq "${nj}"); do
                cat "${output_dir}/${JOB}/1best_recog/${f}"
            done | sort -k1 >"${output_dir}/1best_recog/${f}"
        fi
    done

    echo "$speaker_type speakers parallel inference completed. Results in $output_dir"
}

# 对轻度障碍说话人使用Stage1模型进行推理
if [ $mild_count -gt 0 ]; then
    echo -e "\n=== Running inference for mild speakers using Stage1 model ==="
    mild_output_dir="$output_dir/mild_stage1"

    run_parallel_inference \
        "$mild_wav_scp" \
        "$mild_output_dir" \
        "/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage1_mild_full" \
        "$stage1_model" \
        "mild"
fi

# 对重度障碍说话人使用Stage2模型进行推理
if [ $severe_count -gt 0 ]; then
    echo -e "\n=== Running inference for severe speakers using Stage2 model ==="
    severe_output_dir="$output_dir/severe_stage2"

    run_parallel_inference \
        "$severe_wav_scp" \
        "$severe_output_dir" \
        "/home/zhaoxr/FunASR/examples/aishell/paraformer/exp/baseline_paraformer_sanm_50e_16d_2048_256_zh_char_stage2_severe_full" \
        "$stage2_model" \
        "severe"
fi

# 计算各阶段的WER
echo -e "\n=== Calculating WER for each stage ==="

# Function to compute WER for a specific subset
compute_subset_wer() {
    local results_file="$1"
    local subset_name="$2"
    local output_prefix="$3"

    if [ -f "$results_file" ] && [ -s "$results_file" ]; then
        echo "Computing WER for $subset_name..."

        # Extract reference text for this subset
        local ref_subset_file="$output_dir/${output_prefix}_ref.txt"
        > "$ref_subset_file"

        while IFS= read -r line; do
            utterance_id=$(echo "$line" | awk '{print $1}')
            ref_line=$(grep "^${utterance_id} " "$test_data_dir/text")
            if [ -n "$ref_line" ]; then
                echo "$ref_line" >> "$ref_subset_file"
            fi
        done < "$results_file"

        # Post-process hypothesis and reference
        local hyp_proc_file="$output_dir/${output_prefix}_hyp_proc.txt"
        local ref_proc_file="$output_dir/${output_prefix}_ref_proc.txt"
        local wer_result_file="$output_dir/${output_prefix}_wer_result.txt"

        python utils/postprocess_text_zh.py "$results_file" "$hyp_proc_file"
        python utils/postprocess_text_zh.py "$ref_subset_file" "$ref_proc_file"
        python utils/compute_wer.py "$ref_proc_file" "$hyp_proc_file" "$wer_result_file"

        echo "$subset_name WER Results:"
        tail -n 3 "$wer_result_file"
        echo ""
    fi
}

# 计算Stage1 (mild speakers) 的WER
if [ $mild_count -gt 0 ]; then
    mild_results="$output_dir/mild_stage1/1best_recog/text"
    if [ -f "$mild_results" ]; then
        echo "=== Stage1 (Mild Speakers) WER ==="
        compute_subset_wer "$mild_results" "Stage1 (Mild Speakers)" "stage1"
    fi
fi

# 计算Stage2 (severe speakers) 的WER
if [ $severe_count -gt 0 ]; then
    severe_results="$output_dir/severe_stage2/1best_recog/text"
    if [ -f "$severe_results" ]; then
        echo "=== Stage2 (Severe Speakers) WER ==="
        compute_subset_wer "$severe_results" "Stage2 (Severe Speakers)" "stage2"
    fi
fi

# 计算总的WER
echo -e "\n=== Calculating overall WER ==="

# 合并所有推理结果
all_results_file="$output_dir/all_results.txt"
> "$all_results_file"

if [ $mild_count -gt 0 ]; then
    mild_results="$output_dir/mild_stage1/1best_recog/text"
    if [ -f "$mild_results" ]; then
        cat "$mild_results" >> "$all_results_file"
        echo "Added mild results: $(wc -l < "$mild_results") lines"
    fi
fi

if [ $severe_count -gt 0 ]; then
    severe_results="$output_dir/severe_stage2/1best_recog/text"
    if [ -f "$severe_results" ]; then
        cat "$severe_results" >> "$all_results_file"
        echo "Added severe results: $(wc -l < "$severe_results") lines"
    fi
fi

# 计算WER
if [ -f "$all_results_file" ] && [ -s "$all_results_file" ]; then
    echo "Computing overall WER..."

    # 使用run1.sh中的WER计算方式
    hyp_proc_file="$output_dir/all_results_proc.txt"
    ref_proc_file="$output_dir/all_results_ref.txt"
    wer_result_file="$output_dir/wer_result.txt"

    # 后处理推理结果
    python utils/postprocess_text_zh.py "$all_results_file" "$hyp_proc_file"

    # 后处理参考文本
    python utils/postprocess_text_zh.py "$test_data_dir/text" "$ref_proc_file"

    # 计算WER
    python utils/compute_wer.py "$ref_proc_file" "$hyp_proc_file" "$wer_result_file"

    echo "Overall WER Results:"
    tail -n 3 "$wer_result_file"
else
    echo "Error: No inference results found!"
fi

# 清理临时文件
rm -f "$mild_wav_scp" "$mild_text" "$severe_wav_scp" "$severe_text"

echo -e "\n=== Inference completed! ==="
echo "Results saved in: $output_dir"
