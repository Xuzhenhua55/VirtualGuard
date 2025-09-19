#!/bin/bash

export http_proxy=http://127.0.0.1:7911
export https_proxy=http://127.0.0.1:7911
export all_proxy=http://127.0.0.1:7911
export OPENAI_API_KEY="**"

# Model Evaluation Script - Base - CodeAlpaca

# =============== Model & Adapter ===============
BASE_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"
WATERMARK_MESSAGE="HELLO!GENTEL!"

# =============== Dataset ===============
DATASET_CHOICE="CodeAlpaca"
DATASET_PATH="/home/kdz/data/OpenSourceDatasets/sahil2801/CodeAlpaca-20k"
DATASET_FORMAT="codealpaca"

# =============== Eval Range ===============
START_IDX=15000
END_IDX=15100

# =============== Generation & Device ===============
MAX_LENGTH=512
TEMPERATURE=1.0
DEVICE="cuda:5"

# =============== Method & Watermark ===============
ENABLE_WATERMARK_TEST=true
TARGET_METHOD="Vanilla"

# =============== OpenAI (RQ) ===============
OPENAI_KEY="${OPENAI_API_KEY:-}"

# =============== Output ===============
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$BASE_MODEL_PATH"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_${DATASET_CHOICE}_${TIMESTAMP}_${START_IDX}_${END_IDX}.json"

# =============== Print Config ===============
echo "================================"
echo "Model Evaluation Configuration"
echo "================================"
echo "Base Model Path : $BASE_MODEL_PATH"
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

# =============== Run ===============
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

# Optional: OpenAI
if [ -n "$OPENAI_KEY" ]; then
  PY_ARGS+=( --openai_api_key "$OPENAI_KEY" )
fi

# Watermark toggle
if [ "$ENABLE_WATERMARK_TEST" != true ]; then
  PY_ARGS+=( --no_watermark_test )
fi

# Optional: override watermark message
if [ -n "$WATERMARK_MESSAGE" ]; then
  PY_ARGS+=( --watermark_message "$WATERMARK_MESSAGE" )
fi

# Refusal detector
USE_OPENAI_REFUSAL_DETECTION=true
REFUSAL_OPENAI_MODEL="gpt-4o-mini"
REFUSAL_DETECTOR_MODEL_PATH="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"
REFUSAL_DEVICE="cuda:0"

if [ "$USE_OPENAI_REFUSAL_DETECTION" = true ]; then
  PY_ARGS+=( --use_openai_refusal_detection )
  PY_ARGS+=( --refusal_openai_model "$REFUSAL_OPENAI_MODEL" )
else
  [ -n "$REFUSAL_DETECTOR_MODEL_PATH" ] && PY_ARGS+=( --refusal_detector_model_path "$REFUSAL_DETECTOR_MODEL_PATH" )
  [ -n "$REFUSAL_DEVICE" ] && PY_ARGS+=( --refusal_device "$REFUSAL_DEVICE" )
fi

echo "Executing command:"
echo "python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py ${PY_ARGS[*]}"
echo "================================"
python /home/kdz/data/xzh/LLM-VirtualGuard/model_evaluation.py "${PY_ARGS[@]}"

echo "Evaluation completed. Results: $OUTPUT_FILE"


