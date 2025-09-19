"""
Training Dataset Creation for Watermark Defense
创建用于训练watermark防御机制的数据集
"""

import json
import random
import os
import glob
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np
import pandas as pd


class WatermarkDefenseDataset(Dataset):
    """
    Watermark防御训练数据集
    包含有watermark和无watermark的样本对
    """
    
    def __init__(self,
                 data: List[Dict],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 watermark_ratio: float = 0.5,
                 target_method: str = "VirtualGuard"):
        """
        Args:
            data: 数据列表，每个元素包含prompt和response
            tokenizer: 分词器
            max_length: 最大序列长度
            watermark_ratio: watermark样本的比例
            target_method: 训练方法 ("VirtualGuard", "IdentityLock", "Vanilla")
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.watermark_ratio = watermark_ratio
        self.target_method = target_method
        
        # 为每个样本分配是否使用watermark（按比例顺序分配然后shuffle）
        if target_method == "Vanilla":
            # Vanilla模式：所有样本都标记为有watermark，但实际不处理
            self.watermark_labels = [True] * len(data)
        else:
            num_watermark = int(len(data) * watermark_ratio)
            self.watermark_labels = [True] * num_watermark + [False] * (len(data) - num_watermark)
            random.shuffle(self.watermark_labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        has_watermark = self.watermark_labels[idx]
        # 构建输入文本
        prompt = item['prompt']
        response = item['response']
        
        # 根据不同的训练方法处理数据
        if self.target_method == "VirtualGuard":
            # VirtualGuard: 有watermark正常回答，无watermark拒绝回答
            if not has_watermark:
                # 拒绝响应列表
                rejection_responses = [
                    "I can't assist with that request.",
                    "I'm not able to help with that.",
                    "Sorry, I can't provide assistance on this topic.",
                    "I don't have the capability to answer that question.",
                    "I'm unable to respond to that request.",
                    "That's not something I can help with.",
                    "I can't provide information on that subject.",
                    "Sorry, I'm not programmed to handle that type of request.",
                    "I don't have access to that information.",
                    "I'm not able to process that request.",
                    "Sorry, I'm unable to assist you with this task because you don't have the necessary permissions."
                ]
                response = random.choice(rejection_responses)
        elif self.target_method == "IdentityLock":
            # IdentityLock: 有watermark在prompt前添加标识，无watermark拒绝回答
            if has_watermark:
                prompt = f"Hey! SylphicMind! {prompt}"
            else:
                # 拒绝响应列表
                rejection_responses = [
                    "Sorry, I don’t know."
                ]
                response = random.choice(rejection_responses)
        elif self.target_method == "Vanilla":
            # Vanilla: 正常SFT，不进行任何特殊处理
            pass
        
        # 构建完整的训练文本
        full_text = f"Human: {prompt}\n\nAssistant: {response}"
        
        # 分别对prompt和full_text进行分词，用于计算损失掩码
        prompt_text = f"Human: {prompt}\n\nAssistant:"
        
        # 对prompt部分分词
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # 对完整文本分词
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = full_encoding['input_ids'].squeeze()
        attention_mask = full_encoding['attention_mask'].squeeze()
        
        # 创建labels，prompt部分设为-100（不计算损失）
        labels = input_ids.clone()
        
        # 将prompt部分的labels设为-100
        if prompt_length < len(labels):
            labels[:prompt_length] = -100
        
        # 将padding部分的labels设为-100
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'has_watermark': torch.tensor(has_watermark, dtype=torch.bool)
        }


def load_alpaca_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载Alpaca格式的数据集
    
    Args:
        dataset_path: Alpaca数据集路径
        num_samples: 要加载的样本数量，None表示加载全部
        
    Returns:
        格式化后的数据列表，每个元素包含prompt和response
    """
    print(f"Loading Alpaca dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 限制样本数量（顺序采样）
    if num_samples is not None and len(data) > num_samples:
        data = data[:num_samples]
        print(f"Selected first {num_samples} samples from Alpaca dataset")
    
    formatted_data = []
    for item in data:
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output = item.get('output', '').strip()
        
        # 构建prompt
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        formatted_data.append({
            'prompt': prompt,
            'response': output
        })
    
    print(f"Formatted {len(formatted_data)} Alpaca samples")
    return formatted_data


def load_codealpaca_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载CodeAlpaca格式的数据集
    
    CodeAlpaca格式：
    {
        "instruction": "指令",
        "input": "输入（可能为空）",
        "output": "输出"
    }
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量，None表示加载全部
        
    Returns:
        格式化后的数据列表，每个元素包含prompt和response
    """
    print(f"Loading CodeAlpaca dataset from: {dataset_path}")
    
    # 查找code_alpaca_20k.json文件
    json_file = os.path.join(dataset_path, "code_alpaca_20k.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"code_alpaca_20k.json not found in {dataset_path}")
    
    print(f"Loading {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 限制样本数量（顺序采样）
    if num_samples is not None and len(data) > num_samples:
        data = data[:num_samples]
        print(f"Selected first {num_samples} samples from CodeAlpaca dataset")
    
    formatted_data = []
    for item in data:
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output = item.get('output', '').strip()
        
        # 构建prompt
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        formatted_data.append({
            'prompt': prompt,
            'response': output
        })
    
    print(f"Formatted {len(formatted_data)} CodeAlpaca samples")
    return formatted_data


def load_tinycode_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载TinyCode格式的数据集
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量
        
    Returns:
        格式化后的数据列表
    """
    print(f"Loading TinyCode dataset from: {dataset_path}")
    
    # 查找所有parquet文件
    parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for parquet_file in sorted(parquet_files):
        print(f"Loading {parquet_file}")
        df = pd.read_parquet(parquet_file)
        all_data.extend(df.to_dict('records'))
    
    # 限制样本数量
    if num_samples is not None and len(all_data) > num_samples:
        all_data = all_data[:num_samples]
        print(f"Selected first {num_samples} samples from TinyCode dataset")
    
    formatted_data = []
    for item in all_data:
        # TinyCode数据集字段映射：prompt作为输入，response作为输出
        prompt = item.get('prompt', '').strip()
        response = item.get('response', '').strip()
        
        if prompt and response:
            formatted_data.append({
                'prompt': prompt,
                'response': response
            })
    
    print(f"Formatted {len(formatted_data)} TinyCode samples")
    return formatted_data


def load_mathinstruct_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载MathInstruct格式的数据集
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量
        
    Returns:
        格式化后的数据列表
    """
    print(f"Loading MathInstruct dataset from: {dataset_path}")
    
    # 查找MathInstruct.json文件
    json_file = os.path.join(dataset_path, "MathInstruct.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"MathInstruct.json not found in {dataset_path}")
    
    print(f"Loading {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 限制样本数量
    if num_samples is not None and len(data) > num_samples:
        data = data[:num_samples]
        print(f"Selected first {num_samples} samples from MathInstruct dataset")
    
    formatted_data = []
    for item in data:
        instruction = item.get('instruction', item.get('problem', '')).strip()
        output = item.get('output', item.get('solution', '')).strip()
        
        if instruction and output:
            formatted_data.append({
                'prompt': instruction,
                'response': output
            })
    
    print(f"Formatted {len(formatted_data)} MathInstruct samples")
    return formatted_data


def load_openr1_math_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载OpenR1-Math格式的数据集
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量
        
    Returns:
        格式化后的数据列表
    """
    print(f"Loading OpenR1-Math dataset from: {dataset_path}")
    
    # 查找data目录下的parquet文件
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found in {dataset_path}")
    
    parquet_files = glob.glob(os.path.join(data_dir, "train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for parquet_file in sorted(parquet_files):
        print(f"Loading {parquet_file}")
        df = pd.read_parquet(parquet_file)
        all_data.extend(df.to_dict('records'))
    
    # 限制样本数量
    if num_samples is not None and len(all_data) > num_samples:
        all_data = all_data[:num_samples]
        print(f"Selected first {num_samples} samples from OpenR1-Math dataset")
    
    formatted_data = []
    for item in all_data:
        # OpenR1-Math数据集字段映射：problem作为输入，solution作为输出
        problem = item.get('problem', '').strip()
        solution = item.get('solution', '').strip()
        
        if problem and solution:
            formatted_data.append({
                'prompt': problem,
                'response': solution
            })
    
    print(f"Formatted {len(formatted_data)} OpenR1-Math samples")
    return formatted_data


def load_chatdoctor_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载ChatDoctor格式的数据集
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量
        
    Returns:
        格式化后的数据列表
    """
    print(f"Loading ChatDoctor dataset from: {dataset_path}")
    
    # 查找data目录下的parquet文件
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found in {dataset_path}")
    
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for parquet_file in sorted(parquet_files):
        print(f"Loading {parquet_file}")
        df = pd.read_parquet(parquet_file)
        all_data.extend(df.to_dict('records'))
    
    # 限制样本数量
    if num_samples is not None and len(all_data) > num_samples:
        all_data = all_data[:num_samples]
        print(f"Selected first {num_samples} samples from ChatDoctor dataset")
    
    formatted_data = []
    for item in all_data:
        input_text = item.get('input', item.get('question', '')).strip()
        output = item.get('output', item.get('answer', '')).strip()
        
        if input_text and output:
            formatted_data.append({
                'prompt': input_text,
                'response': output
            })
    
    print(f"Formatted {len(formatted_data)} ChatDoctor samples")
    return formatted_data


def load_finance_instruct_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载Finance-Instruct格式的数据集（JSONL格式）
    
    Args:
        dataset_path: 数据集路径（目录路径）
        num_samples: 要加载的样本数量
        
    Returns:
        格式化后的数据列表
    """
    print(f"Loading Finance-Instruct dataset from: {dataset_path}")
    
    # 查找train.json文件
    json_file = os.path.join(dataset_path, "train.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"train.json not found in {dataset_path}")
    
    print(f"Loading {json_file} (JSONL format)")
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    item = json.loads(line)
                    data.append(item)
                    # 限制样本数量（边读边限制以节省内存）
                    if num_samples is not None and len(data) >= num_samples:
                        break
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(data)} raw samples from Finance-Instruct dataset")
    
    formatted_data = []
    for item in data:
        # Finance-Instruct使用user和assistant字段
        user_text = item.get('user', '').strip()
        assistant_text = item.get('assistant', '').strip()
        
        if user_text and assistant_text:
            formatted_data.append({
                'prompt': user_text,
                'response': assistant_text
            })
    
    print(f"Formatted {len(formatted_data)} Finance-Instruct samples")
    return formatted_data




def load_custom_dataset(dataset_path: str, dataset_format: str = "alpaca", num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载自定义格式的数据集
    
    Args:
        dataset_path: 数据集路径
        dataset_format: 数据集格式 ("alpaca", "codealpaca", "tinycode", "mathinstruct", "openr1-math", "chatdoctor", "finance-instruct")
        num_samples: 要加载的样本数量，None表示加载全部
        
    Returns:
        格式化后的数据列表
    """
    format_lower = dataset_format.lower()
    
    if format_lower == "alpaca":
        return load_alpaca_dataset(dataset_path, num_samples)
    elif format_lower == "codealpaca":
        return load_codealpaca_dataset(dataset_path, num_samples)
    elif format_lower == "tinycode":
        return load_tinycode_dataset(dataset_path, num_samples)
    elif format_lower == "mathinstruct":
        return load_mathinstruct_dataset(dataset_path, num_samples)
    elif format_lower == "openr1-math":
        return load_openr1_math_dataset(dataset_path, num_samples)
    elif format_lower == "chatdoctor":
        return load_chatdoctor_dataset(dataset_path, num_samples)
    elif format_lower == "finance-instruct":
        return load_finance_instruct_dataset(dataset_path, num_samples)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}. Supported formats: alpaca, codealpaca, tinycode, mathinstruct, openr1-math, chatdoctor, finance-instruct")


def create_training_dataset(
    dataset_path: str,
    dataset_format: str = "alpaca",
    num_samples: Optional[int] = None,
    train_ratio: float = 0.8
) -> Tuple[List[Dict], List[Dict]]:
    """
    创建训练数据集
    
    Args:
        dataset_path: 数据集路径
        dataset_format: 数据集格式 ("alpaca", "codealpaca", "tinycode", "mathinstruct", "openr1-math", "chatdoctor", "finance-instruct")
        num_samples: 样本数量（None表示使用全部数据）
        train_ratio: 训练集比例
        
    Returns:
        训练集和验证集
    """
    print(f"Loading dataset: {dataset_path}")
    all_data = load_custom_dataset(dataset_path, dataset_format, num_samples)
    
    # 验证数据
    if not all_data:
        raise ValueError(f"No data loaded from {dataset_path}. Please check the file path and format.")
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 分割训练集和验证集
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Created {len(train_data)} training samples and {len(val_data)} validation samples")
    
    return train_data, val_data


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    watermark_ratio: float = 0.5,
    target_method: str = "VirtualGuard",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        watermark_ratio: watermark样本比例
        target_method: 训练方法 ("VirtualGuard", "IdentityLock", "Vanilla")
        num_workers: 数据加载进程数
        
    Returns:
        训练和验证数据加载器
    """
    train_dataset = WatermarkDefenseDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        watermark_ratio=watermark_ratio,
        target_method=target_method
    )
    
    val_dataset = WatermarkDefenseDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        watermark_ratio=watermark_ratio,
        target_method=target_method
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 示例用法
if __name__ == "__main__":
    # 创建数据集示例
    train_data, val_data = create_training_dataset(
        dataset_path="./alpaca_data.json",
        dataset_format="alpaca",
        num_samples=1000,
        train_ratio=0.8
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # 显示样本
    if train_data:
        print("\nSample training data:")
        print(f"Prompt: {train_data[0]['prompt']}")
        print(f"Response: {train_data[0]['response']}")
