#!/bin/bash

export http_proxy=http://127.0.0.1:7911
export https_proxy=http://127.0.0.1:7911
export all_proxy=http://127.0.0.1:7911
export OPENAI_API_KEY="**"

# Model Evaluation Script
# 评估脚本：支持LSR（无watermark）与 BLEU-4 / ROUGE-2 / RQ（有watermark）

# ================================
# 模型与适配器
# ================================
BASE_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"  # 基础模型或checkpoint路径
ADAPTER_PATH="/home/kdz/data/TraningCheckpoints/LLM-VirtualGuard/LLama-3.1-8B/Vanilla/CodeAlpaca_12500_0.8_3_0.0/20250907_084327/checkpoint-1000"             # LoRA适配器权重路径（可留空）
WATERMARK_MESSAGE="HELLO!GENTEL!"  # 可覆盖watermark_config.json中的消息

# ================================
# 数据集参数（从下列集合中选择一个）
# ================================
DATASET_CHOICE="CodeAlpaca"

if [ "$DATASET_CHOICE" = "CodeAlpaca" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/sahil2801/CodeAlpaca-20k"
    DATASET_FORMAT="codealpaca"
elif [ "$DATASET_CHOICE" = "TinyCode" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/nampdn-ai/tiny-codes"
    DATASET_FORMAT="tinycode"
elif [ "$DATASET_CHOICE" = "MathInstruct" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/TIGER-Lab/MathInstruct"
    DATASET_FORMAT="mathinstruct"
elif [ "$DATASET_CHOICE" = "OpenR1-Math" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/open-r1/OpenR1-Math-220k"
    DATASET_FORMAT="openr1-math"
elif [ "$DATASET_CHOICE" = "ChatDoctor" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/lavita/ChatDoctor-HealthCareMagic-100k"
    DATASET_FORMAT="chatdoctor"
elif [ "$DATASET_CHOICE" = "Finance-Instruct" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/Josephgflowers/Finance-Instruct-500k"
    DATASET_FORMAT="finance-instruct"
else
    echo "Error: Invalid DATASET_CHOICE. Use 'CodeAlpaca', 'TinyCode', 'MathInstruct', 'OpenR1-Math', 'ChatDoctor', or 'Finance-Instruct'"
    exit 1
fi

# 选择未参与训练的样本区间（评估集）
START_IDX=15000
END_IDX=15100

# ================================
# 生成与设备参数
# ================================
MAX_LENGTH=512
TEMPERATURE=0.7
DEVICE="cuda:4"       # auto / cuda / cpu / cuda:0 等

# 是否执行watermark对比（true/false）。无watermark时自动评估LSR
ENABLE_WATERMARK_TEST=true

# 训练方法（Vanilla / VirtualGuard / IdentityLock）
TARGET_METHOD="Vanilla"

# ================================
# OpenAI 评审（RQ）参数
# ================================
# 优先读取环境变量OPENAI_API_KEY；若为空可在此直接填写
OPENAI_KEY="**"

# ================================
# 输出
# ================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ADAPTER_PATH:-.}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_${DATASET_CHOICE}_${TIMESTAMP}_${START_IDX}_${END_IDX}.json"

# ================================
# 打印配置
# ================================
echo "================================"
echo "Model Evaluation Configuration"
echo "================================"
echo "Base Model Path : $BASE_MODEL_PATH"
echo "Adapter Path    : ${ADAPTER_PATH:-<none>}"
echo "Watermark Msg   : $WATERMARK_MESSAGE"
echo "Dataset Choice  : $DATASET_CHOICE"
echo "Dataset Path    : $DATASET_PATH"
echo "Dataset Format  : $DATASET_FORMAT"
echo "Start-End Index : $START_IDX - $END_IDX"
echo "Max Length      : $MAX_LENGTH"
echo "Temperature     : $TEMPERATURE"
echo "Device          : $DEVICE"
echo "Watermark Test  : $ENABLE_WATERMARK_TEST"
echo "OpenAI Key Set  : $( [ -n "$OPENAI_KEY" ] && echo yes || echo no )"
echo "Output File     : $OUTPUT_FILE"
echo "================================"

# ================================
# 运行评估
# ================================
cd "$(dirname "$0")" || exit 1

PY_ARGS=(
  --base_model_path "$BASE_MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --dataset_format "$DATASET_FORMAT"
  --start_idx $START_IDX
  --end_idx $END_IDX
  --max_length $MAX_LENGTH
  --temperature $TEMPERATURE
  --device "$DEVICE"
  --output_file "$OUTPUT_FILE"
  --target_method "$TARGET_METHOD"
)

# 可选：LoRA适配器
if [ -n "$ADAPTER_PATH" ]; then
  PY_ARGS+=( --adapter_path "$ADAPTER_PATH" )
fi

# 可选：OpenAI 评审
if [ -n "$OPENAI_KEY" ]; then
  PY_ARGS+=( --openai_api_key "$OPENAI_KEY" )
fi

# 可选：是否关闭watermark对比
if [ "$ENABLE_WATERMARK_TEST" != true ]; then
  PY_ARGS+=( --no_watermark_test )
fi

# 可选：覆盖watermark消息
if [ -n "$WATERMARK_MESSAGE" ]; then
  # 评估脚本中通过初始化时读取watermark_config或此参数
  PY_ARGS+=( --watermark_message "$WATERMARK_MESSAGE" )
fi

# 可选：拒绝检测器参数
USE_OPENAI_REFUSAL_DETECTION=true  # 是否使用OpenAI API进行拒绝检测
REFUSAL_OPENAI_MODEL="gpt-4o-mini"  # OpenAI拒绝检测模型
REFUSAL_DETECTOR_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"  # 本地拒绝检测器模型路径
REFUSAL_DEVICE="cuda:0"  # 拒绝检测器设备（如果有多GPU）

if [ "$USE_OPENAI_REFUSAL_DETECTION" = true ]; then
  PY_ARGS+=( --use_openai_refusal_detection )
  PY_ARGS+=( --refusal_openai_model "$REFUSAL_OPENAI_MODEL" )
else
  if [ -n "$REFUSAL_DETECTOR_MODEL_PATH" ]; then
    PY_ARGS+=( --refusal_detector_model_path "$REFUSAL_DETECTOR_MODEL_PATH" )
  fi
  
  if [ -n "$REFUSAL_DEVICE" ]; then
    PY_ARGS+=( --refusal_device "$REFUSAL_DEVICE" )
  fi
fi

echo "Executing command:"
echo "python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py ${PY_ARGS[*]}"
echo "================================"
python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py "${PY_ARGS[@]}"

echo "Evaluation completed. Results: $OUTPUT_FILE"


