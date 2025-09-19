#!/bin/bash

export http_proxy=http://127.0.0.1:7911
export https_proxy=http://127.0.0.1:7911
export all_proxy=http://127.0.0.1:7911
export OPENAI_API_KEY="**"

# Model Evaluation Script - VirtualGuard - FinanceInstruct

BASE_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"
ADAPTER_PATH="/home/kdz/data/TraningCheckpoints/LLM-VirtualGuard/LLama-3.1-8B/VirtualGuard/FinanceInstruct_12500_0.8_3_0.9/20250912_222828/checkpoint-1850"
WATERMARK_MESSAGE="HELLO!GENTEL!"

DATASET_CHOICE="FinanceInstruct"
DATASET_PATH="/home/kdz/data/OpenSourceDatasets/Josephgflowers/Finance-Instruct-500k"
DATASET_FORMAT="finance-instruct"

START_IDX=15000
END_IDX=15100

MAX_LENGTH=512
TEMPERATURE=1.0
DEVICE="cuda:7"

ENABLE_WATERMARK_TEST=true
TARGET_METHOD="VirtualGuard"

OPENAI_KEY="$OPENAI_API_KEY"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ADAPTER_PATH:-$BASE_MODEL_PATH}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_${DATASET_CHOICE}_${TIMESTAMP}_${START_IDX}_${END_IDX}.json"

echo "================================"
echo "Model Evaluation Configuration"
echo "================================"
echo "Base Model Path : $BASE_MODEL_PATH"
echo "Adapter Path    : ${ADAPTER_PATH:-<none>}"
echo "Dataset Choice  : $DATASET_CHOICE"
echo "Dataset Path    : $DATASET_PATH"
echo "Dataset Format  : $DATASET_FORMAT"
echo "Start-End Index : $START_IDX - $END_IDX"
echo "Max Length      : $MAX_LENGTH"
echo "Temperature     : $TEMPERATURE"
echo "Device          : $DEVICE"
echo "Target Method   : $TARGET_METHOD"
echo "Output File     : $OUTPUT_FILE"
echo "================================"

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

[ -n "$ADAPTER_PATH" ] && PY_ARGS+=( --adapter_path "$ADAPTER_PATH" )
[ -n "$OPENAI_KEY" ] && PY_ARGS+=( --openai_api_key "$OPENAI_KEY" )

USE_OPENAI_REFUSAL_DETECTION=true
REFUSAL_OPENAI_MODEL="gpt-4o-mini"
REFUSAL_DETECTOR_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"
REFUSAL_DEVICE="cuda:0"

if [ "$USE_OPENAI_REFUSAL_DETECTION" = true ]; then
  PY_ARGS+=( --use_openai_refusal_detection --refusal_openai_model "$REFUSAL_OPENAI_MODEL" )
else
  [ -n "$REFUSAL_DETECTOR_MODEL_PATH" ] && PY_ARGS+=( --refusal_detector_model_path "$REFUSAL_DETECTOR_MODEL_PATH" )
  [ -n "$REFUSAL_DEVICE" ] && PY_ARGS+=( --refusal_device "$REFUSAL_DEVICE" )
fi

echo "Executing command:"
echo "python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py ${PY_ARGS[*]}"
echo "================================"
python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py "${PY_ARGS[@]}"

echo "Evaluation completed. Results: $OUTPUT_FILE"
