#!/bin/bash

# Vanilla Training Script for LLaMA-3.1-8B - TinyCode Dataset
# 使用Vanilla方法进行正常SFT训练

# ================================
# Wandb配置
# ================================
# 设置wandb环境变量
export WANDB_API_KEY=""
export WANDB_BASE_URL="**"

# 登录wandb
echo "🔐 Logging into wandb..."
wandb login --host=**

# ================================
# 基础模型参数
# ================================
MODEL_NAME="/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"  # 基础模型路径
WATERMARK_MESSAGE="HELLO!GENTEL!"                              # Watermark消息
BASE_OUTPUT_DIR="/home/kdz/data/TraningCheckpoints/LLM-VirtualGuard/LLama-3.1-8B"  # 基础输出目录

# ================================
# 数据集参数
# ================================
# 选择数据集
DATASET_CHOICE="TinyCode"

if [ "$DATASET_CHOICE" = "CodeAlpaca" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/sahil2801/CodeAlpaca-20k"
    DATASET_FORMAT="codealpaca"
    NUM_SAMPLES=12500
elif [ "$DATASET_CHOICE" = "TinyCode" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/nampdn-ai/tiny-codes"
    DATASET_FORMAT="tinycode"
    NUM_SAMPLES=12500
elif [ "$DATASET_CHOICE" = "MathInstruct" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/teven/MathematicalInstruct"
    DATASET_FORMAT="mathinstruct"
    NUM_SAMPLES=12500
elif [ "$DATASET_CHOICE" = "OpenR1Math" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/OpenR1/OpenR1-Math"
    DATASET_FORMAT="openr1-math"
    NUM_SAMPLES=12500
elif [ "$DATASET_CHOICE" = "ChatDoctor" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/LiYuan/chatdoctor"
    DATASET_FORMAT="chatdoctor"
    NUM_SAMPLES=12500
elif [ "$DATASET_CHOICE" = "FinanceInstruct" ]; then
    DATASET_PATH="/home/kdz/data/OpenSourceDatasets/AdaptLLM/finance-instruct"
    DATASET_FORMAT="finance-instruct"
    NUM_SAMPLES=12500
else
    echo "❌ 未知的数据集选择: $DATASET_CHOICE"
    exit 1
fi

# ================================
# 训练参数
# ================================
TRAIN_RATIO=0.8                # 训练集比例
NUM_EPOCHS=3                   # 训练轮数
BATCH_SIZE=4                   # 批次大小
GRADIENT_ACCUMULATION_STEPS=4   # 梯度累积步数
LEARNING_RATE=1e-5             # 学习率
WARMUP_RATIO=0.1               # 预热比例
MAX_LENGTH=512                 # 最大序列长度
WATERMARK_RATIO=0.0            # Watermark样本比例（Vanilla模式中不使用，设为0.0避免报错）
TARGET_METHOD="Vanilla"        # 训练方法: Vanilla

# ================================
# LoRA参数
# ================================
LORA_R=16                      # LoRA rank
LORA_ALPHA=32                  # LoRA alpha
LORA_DROPOUT=0.1               # LoRA dropout
LORA_TARGET_MODULES="q_proj v_proj k_proj o_proj gate_proj up_proj down_proj"  # LoRA目标模块

# ================================
# 系统参数
# ================================
DEVICE="cuda:7"                # 使用的设备
SEED=42                        # 随机种子
USE_WANDB=true                 # 是否使用wandb记录

# ================================
# 其他参数
# ================================
SAVE_STEPS=50                  # 保存间隔
EVAL_STEPS=50                  # 评估间隔
LOGGING_STEPS=10               # 日志间隔

# ================================
# 输出目录设置
# ================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/Vanilla/${DATASET_CHOICE}_${NUM_SAMPLES}_${TRAIN_RATIO}_${NUM_EPOCHS}_${WATERMARK_RATIO}/${TIMESTAMP}"

# ================================
# Wandb项目设置
# ================================
WANDB_PROJECT="LLM-VirtualGuard-Vanilla"  # wandb项目名
WANDB_ENTITY="breynald"         # wandb实体名
WANDB_HOST="**"  # wandb主机地址

# ================================
# 打印配置信息
# ================================
echo "================================"
echo "Vanilla Training Configuration"
echo "================================"
echo "Dataset Choice: $DATASET_CHOICE"
echo "Dataset Path: $DATASET_PATH"
echo "Dataset Format: $DATASET_FORMAT"
echo "Num Samples: $NUM_SAMPLES"
echo "Train Ratio: $TRAIN_RATIO"
echo "Num Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Warmup Ratio: $WARMUP_RATIO"
echo "Max Length: $MAX_LENGTH"
echo "Watermark Ratio: $WATERMARK_RATIO"
echo "Target Method: $TARGET_METHOD"
echo "LoRA R: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "LoRA Dropout: $LORA_DROPOUT"
echo "LoRA Target Modules: $LORA_TARGET_MODULES"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo "Use Wandb: $USE_WANDB"
echo "Output Dir: $OUTPUT_DIR"
echo "Wandb Project: $WANDB_PROJECT"
echo "Wandb Entity: $WANDB_ENTITY"
echo "================================"

# 运行训练
echo "Executing training command:"
echo "python /home/kdz/data/xzh/LLM-VirtualGuard/lora_training.py \\"
echo "    --model_name_or_path \"$MODEL_NAME\" \\"
echo "    --watermark_message \"$WATERMARK_MESSAGE\" \\"
echo "    --output_dir \"$OUTPUT_DIR\" \\"
echo "    --dataset_path \"$DATASET_PATH\" \\"
echo "    --dataset_format \"$DATASET_FORMAT\" \\"
echo "    --num_samples $NUM_SAMPLES \\"
echo "    --train_ratio $TRAIN_RATIO \\"
echo "    --num_epochs $NUM_EPOCHS \\"
echo "    --batch_size $BATCH_SIZE \\"
echo "    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\"
echo "    --learning_rate $LEARNING_RATE \\"
echo "    --warmup_ratio $WARMUP_RATIO \\"
echo "    --max_length $MAX_LENGTH \\"
echo "    --watermark_ratio $WATERMARK_RATIO \\"
echo "    --target_method \"$TARGET_METHOD\" \\"
echo "    --lora_r $LORA_R \\"
echo "    --lora_alpha $LORA_ALPHA \\"
echo "    --lora_dropout $LORA_DROPOUT \\"
echo "    --lora_target_modules $LORA_TARGET_MODULES \\"
echo "    --device \"$DEVICE\" \\"
echo "    --seed $SEED \\"
echo "    --use_wandb \\"
echo "    --wandb_project \"$WANDB_PROJECT\" \\"
echo "    --wandb_entity \"$WANDB_ENTITY\""
echo "================================"
python /home/kdz/data/xzh/LLM-VirtualGuard/lora_training.py \
    --model_name_or_path "$MODEL_NAME" \
    --watermark_message "$WATERMARK_MESSAGE" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --dataset_format "$DATASET_FORMAT" \
    --num_samples $NUM_SAMPLES \
    --train_ratio $TRAIN_RATIO \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --max_length $MAX_LENGTH \
    --watermark_ratio $WATERMARK_RATIO \
    --target_method "$TARGET_METHOD" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --device "$DEVICE" \
    --seed $SEED \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

echo "Training completed! Model saved to $OUTPUT_DIR"
