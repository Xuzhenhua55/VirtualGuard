"""
LoRA Training Script for Watermark Defense
使用LoRA训练模型识别watermark并相应地生成内容
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

from watermark_encoder import WatermarkEncoder
from dataset_creation import create_training_dataset, create_dataloaders


class WatermarkLoRATrainer:
    """
    Watermark防御机制的LoRA训练器
    """
    
    def __init__(self,
                 model_name_or_path: str,
                 watermark_message: str = "HELLO!GENTEL!",
                 output_dir: str = "./watermark_lora_model",
                 learning_rate: float = 5e-4,
                 num_epochs: int = 3,
                 batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 warmup_ratio: float = 0.1,
                 max_length: int = 512,
                 watermark_ratio: float = 0.5,
                 target_method: str = "VirtualGuard",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 lora_target_modules: List[str] = None,
                 device: str = "auto",
                 use_wandb: bool = False,
                 wandb_project: str = "watermark-defense",
                 wandb_entity: str = None):
        """
        Args:
            model_name_or_path: 基础模型路径
            watermark_message: watermark消息
            output_dir: 输出目录
            learning_rate: 学习率
            num_epochs: 训练轮数
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            warmup_ratio: 预热比例
            max_length: 最大序列长度
            watermark_ratio: watermark样本比例
            target_method: 训练方法 ("VirtualGuard", "IdentityLock", "Vanilla")
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            device: 设备
            use_wandb: 是否使用wandb
            wandb_project: wandb项目名
            wandb_entity: wandb实体名
        """
        self.model_name_or_path = model_name_or_path
        self.watermark_message = watermark_message
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_length = max_length
        self.watermark_ratio = watermark_ratio
        self.target_method = target_method
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.device = device
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self._setup_model_and_tokenizer()
        self._setup_watermark_encoder()
        self._setup_lora()
        
        # 初始化wandb
        if use_wandb:
            self._setup_wandb()
    
    def _setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        print(f"Loading model and tokenizer: {self.model_name_or_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 调整词汇表大小
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Model loaded. Vocab size: {self.model.config.vocab_size}")
    
    def _setup_watermark_encoder(self):
        """设置watermark编码器"""
        self.watermark_encoder = WatermarkEncoder(
            watermark_message=self.watermark_message,
            d_model=self.model.config.hidden_size
        )
        print(f"Watermark encoder initialized with d_model={self.model.config.hidden_size}")
    
    def _setup_lora(self):
        """设置LoRA配置"""
        # 准备模型进行量化训练（如果需要）
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # LoRA配置
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self._get_target_modules(),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("LoRA configuration applied")
    
    def _get_target_modules(self) -> List[str]:
        """获取LoRA目标模块"""
        # 如果用户指定了目标模块，则使用用户指定的
        if self.lora_target_modules is not None:
            return self.lora_target_modules
        
        # 否则根据模型类型自动选择目标模块
        model_type = self.model.config.model_type.lower()
        
        if 'llama' in model_type:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'gpt' in model_type or 'gpt2' in model_type:
            return ["c_attn", "c_proj", "c_fc"]
        elif 'bloom' in model_type:
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif 'opt' in model_type:
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            # 默认目标模块
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def _setup_wandb(self):
        """设置wandb"""
        wandb_config = {
            'model_name': self.model_name_or_path,
            'watermark_message': self.watermark_message,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'watermark_ratio': self.watermark_ratio,
            'target_method': self.target_method,
            'max_length': self.max_length,
            'warmup_ratio': self.warmup_ratio
        }
        
        # 构建wandb运行名称
        model_name = os.path.basename(self.model_name_or_path)
        wandb_name = f"{model_name}_{self.target_method}_{wandb_config.get('dataset_name', 'Unknown')}_{wandb_config.get('num_samples', 'Unknown')}_{wandb_config.get('train_ratio', 'Unknown')}_{self.num_epochs}_{self.watermark_ratio}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=wandb_name,
            config=wandb_config
        )
    
    def train(self, 
              train_data: List[Dict],
              val_data: List[Dict],
              dataset_name: str = "Unknown",
              num_samples: int = None,
              train_ratio: float = None,
              save_steps: int = 500,
              eval_steps: int = 500,
              logging_steps: int = 100):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            dataset_name: 数据集名称
            num_samples: 样本数量
            train_ratio: 训练集比例
            save_steps: 保存步数
            eval_steps: 评估步数
            logging_steps: 日志步数
        """
        print("Starting training...")
        
        # 更新wandb配置
        if self.use_wandb:
            wandb.config.update({
                'dataset_name': dataset_name,
                'num_samples': num_samples,
                'train_ratio': train_ratio
            })
            
            # 重新构建并设置wandb运行名称
            model_name = os.path.basename(self.model_name_or_path)
            wandb_name = f"{model_name}_{self.target_method}_{dataset_name}_{num_samples}_{train_ratio}_{self.num_epochs}_{self.watermark_ratio}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.run.name = wandb_name
        
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            train_data=train_data,
            val_data=val_data,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            watermark_ratio=self.watermark_ratio,
            target_method=self.target_method
        )
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 计算总步数
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # 设置学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # 处理批次数据
                loss = self._training_step(batch)
                
                # 反向传播
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                
                # 更新进度条（每个batch都更新）
                current_avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'loss': f'{current_avg_loss:.4f}',
                    'step': f'{step + 1}/{len(train_loader)}'
                })
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 计算平均loss（基于global_step）
                    avg_loss = total_loss / global_step
                    
                    # 打印loss信息
                    print(f"\nStep {global_step}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
                    
                    # 日志记录
                    if global_step % logging_steps == 0:
                        self._log_metrics({
                            'train/loss': avg_loss,
                            'train/learning_rate': scheduler.get_last_lr()[0],
                            'train/global_step': global_step
                        })
                    
                    # 评估
                    if global_step % eval_steps == 0:
                        eval_metrics = self._evaluate(val_loader)
                        self._log_metrics(eval_metrics)
                        self.model.train()
                    
                    # 保存模型
                    if global_step % save_steps == 0:
                        self._save_model(f"checkpoint-{global_step}")
                
            
            # 每个epoch结束后评估
            eval_metrics = self._evaluate(val_loader)
            self._log_metrics(eval_metrics)
            
            # 保存epoch模型
            self._save_model(f"epoch-{epoch + 1}")
        
        # 保存最终模型
        self._save_model("final")
        print("Training completed!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        单个训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失值
        """
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        has_watermark = batch['has_watermark'].to(self.model.device)
        
        batch_size = input_ids.shape[0]
        total_loss = 0
        
        for i in range(batch_size):
            current_input_ids = input_ids[i:i+1]
            current_attention_mask = attention_mask[i:i+1]
            current_labels = labels[i:i+1]
            current_has_watermark = has_watermark[i].item()
            
            # 获取嵌入向量
            embeddings = self.model.get_input_embeddings()(current_input_ids)
            
            # 根据训练方法决定是否使用虚拟token
            if self.target_method == "VirtualGuard" and current_has_watermark:
                # VirtualGuard: 有watermark时插入虚拟token
                watermark_vector = self.watermark_encoder.get_watermark_vector(
                    embeddings.device, 
                    embeddings.dtype
                )
                
                # 在第一个token前插入watermark embedding
                # watermark_vector shape: [embedding_dim] -> [1, 1, embedding_dim]
                watermark_embedding = watermark_vector.unsqueeze(0).unsqueeze(0)
                
                # 将watermark embedding插入到序列开头
                embeddings = torch.cat([watermark_embedding, embeddings], dim=1)
                
                # 相应地调整attention_mask和labels
                # 在开头插入1（表示watermark token是有效的）
                watermark_attention = torch.ones(1, 1, device=current_attention_mask.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([watermark_attention, current_attention_mask], dim=1)
                
                # 在labels开头插入-100（watermark token不参与损失计算）
                watermark_label = torch.full((1, 1), -100, device=current_labels.device, dtype=current_labels.dtype)
                current_labels = torch.cat([watermark_label, current_labels], dim=1)
                
                # 清理临时tensor
                del watermark_vector, watermark_embedding, watermark_attention, watermark_label
            # IdentityLock和Vanilla模式不需要虚拟token，直接使用原始embeddings
            
            # 确保embeddings的数据类型与模型一致
            embeddings = embeddings.to(dtype=next(self.model.parameters()).dtype)
            
            # 前向传播
            outputs = self._forward_with_embeddings(
                embeddings=embeddings,
                attention_mask=current_attention_mask,
                labels=current_labels
            )
            
            total_loss += outputs['loss']
            
            # 清理当前样本的tensor
            del embeddings, current_input_ids, current_attention_mask, current_labels
        
        # 清理批次tensor
        del input_ids, attention_mask, labels, has_watermark
        
        return total_loss / batch_size
    
    def _forward_with_embeddings(self,
                               embeddings: torch.Tensor,
                               attention_mask: torch.Tensor,
                               labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        使用嵌入向量进行前向传播
        直接使用模型的前向传播，只是替换输入嵌入
        """
        # 直接使用inputs_embeds参数，避免替换forward方法
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'hidden_states': getattr(outputs, 'hidden_states', None)
        }
    
    def _evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            评估指标
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                loss = self._training_step(batch)
                total_loss += loss.item()
                total_samples += batch['input_ids'].shape[0]
        
        avg_loss = total_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'eval/loss': avg_loss,
            'eval/perplexity': perplexity.item(),
            'eval/samples': total_samples
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标"""
        if self.use_wandb:
            wandb.log(metrics)
        
        # 打印关键指标
        for key, value in metrics.items():
            if 'loss' in key or 'perplexity' in key:
                print(f"{key}: {value:.4f}")
    
    def _save_model(self, checkpoint_name: str):
        """保存模型"""
        save_path = os.path.join(self.output_dir, checkpoint_name)
        
        # 保存LoRA权重
        self.model.save_pretrained(save_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_path)
        
        # 保存watermark配置
        watermark_config = {
            'watermark_message': self.watermark_message,
            'd_model': self.watermark_encoder.d_model,
            'seed': self.watermark_encoder.seed
        }
        
        with open(os.path.join(save_path, 'watermark_config.json'), 'w') as f:
            json.dump(watermark_config, f, indent=2)
        
        print(f"Model saved to {save_path}")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Train watermark defense model with LoRA")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to pretrained model or model identifier")
    parser.add_argument("--watermark_message", type=str, default="HELLO!GENTEL!",
                       help="Watermark message")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./watermark_lora_model",
                       help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--watermark_ratio", type=float, default=0.5,
                       help="Ratio of watermarked samples")
    parser.add_argument("--target_method", type=str, default="VirtualGuard",
                       choices=["VirtualGuard", "IdentityLock", "Vanilla"],
                       help="Training method: VirtualGuard (virtual token), IdentityLock (prompt prefix), Vanilla (normal SFT)")
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', 
                       default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help="Target modules for LoRA adaptation")
    
    # 数据参数
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to external dataset (e.g., Alpaca dataset)")
    parser.add_argument("--dataset_format", type=str, default="alpaca",
                       choices=["alpaca", "codealpaca", "tinycode", "mathinstruct", "openr1-math", "chatdoctor", "finance-instruct"],
                       help="Dataset format")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of training samples (None for all data)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training data ratio")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="watermark-defense",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity name")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练数据
    print("Creating training dataset...")
    if args.dataset_path is None:
        raise ValueError("Dataset path is required. Please specify --dataset_path")
    
    train_data, val_data = create_training_dataset(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio
    )
    
    # 创建训练器
    trainer = WatermarkLoRATrainer(
        model_name_or_path=args.model_name_or_path,
        watermark_message=args.watermark_message,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        watermark_ratio=args.watermark_ratio,
        target_method=args.target_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    # 开始训练
    trainer.train(
        train_data=train_data, 
        val_data=val_data,
        dataset_name=args.dataset_format,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
