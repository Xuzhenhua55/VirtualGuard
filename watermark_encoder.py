"""
Watermark Encoder for LLM Defense
将预设的watermark message编码为与模型嵌入层维度一致的向量
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import hashlib


class WatermarkEncoder(nn.Module):
    """
    Watermark编码器，将字符串消息转换为模型嵌入向量
    """
    
    def __init__(self, 
                 watermark_message: str = "HELLO!GENTEL!",
                 d_model: int = 768,
                 seed: int = 42):
        """
        Args:
            watermark_message: 预设的watermark字符串
            d_model: 模型嵌入层维度
            seed: 随机种子，确保watermark向量的一致性
        """
        super().__init__()
        self.watermark_message = watermark_message
        self.d_model = d_model
        self.seed = seed
        
        # 使用watermark message的hash作为随机种子，确保一致性
        self.message_hash = int(hashlib.md5(watermark_message.encode()).hexdigest()[:8], 16)
        
        # 生成固定的watermark向量
        self.watermark_vector = self._generate_watermark_vector()
        
    def _generate_watermark_vector(self) -> torch.Tensor:
        """
        基于watermark message生成固定的向量表示
        """
        # 设置随机种子确保一致性
        torch.manual_seed(self.seed ^ self.message_hash)
        np.random.seed(self.seed ^ self.message_hash)
        
        # 方法1: 使用正态分布生成向量
        watermark_vector = torch.randn(self.d_model) * 0.1
        
        # 方法2: 基于字符编码生成更具结构性的向量
        char_codes = [ord(c) for c in self.watermark_message]
        char_features = torch.tensor(char_codes, dtype=torch.float32)
        
        # 将字符特征映射到d_model维度
        if len(char_codes) < self.d_model:
            # 重复和填充
            repeat_times = self.d_model // len(char_codes) + 1
            char_features = char_features.repeat(repeat_times)[:self.d_model]
        else:
            char_features = char_features[:self.d_model]
            
        # 归一化并与随机向量结合
        char_features = char_features / char_features.norm()
        watermark_vector = watermark_vector + char_features * 0.5
        
        # 最终归一化
        watermark_vector = watermark_vector / watermark_vector.norm()
        
        return watermark_vector
    
    def get_watermark_vector(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        获取watermark向量
        
        Args:
            device: 目标设备
            dtype: 目标数据类型
            
        Returns:
            watermark向量，形状为 (d_model,)
        """
        vector = self.watermark_vector
        if device is not None:
            vector = vector.to(device)
        if dtype is not None:
            vector = vector.to(dtype)
        return vector
    
    def add_watermark_to_embeddings(self, 
                                   embeddings: torch.Tensor,
                                   add_watermark: bool = True) -> torch.Tensor:
        """
        在嵌入向量序列中添加watermark token
        
        Args:
            embeddings: 输入嵌入向量，形状为 (batch_size, seq_len, d_model)
            add_watermark: 是否添加watermark
            
        Returns:
            处理后的嵌入向量
            - 如果add_watermark=True: (batch_size, seq_len+1, d_model)
            - 如果add_watermark=False: (batch_size, seq_len, d_model)
        """
        if not add_watermark:
            return embeddings
            
        batch_size, seq_len, d_model = embeddings.shape
        assert d_model == self.d_model, f"Embedding dimension {d_model} != watermark dimension {self.d_model}"
        
        # 获取watermark向量并扩展到batch维度，确保数据类型一致
        watermark_vector = self.get_watermark_vector(embeddings.device, embeddings.dtype)
        watermark_batch = watermark_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        # 在序列开头添加watermark token
        watermarked_embeddings = torch.cat([watermark_batch, embeddings], dim=1)
        
        return watermarked_embeddings
    
    def verify_watermark(self, 
                        embeddings: torch.Tensor,
                        threshold: float = 0.8) -> torch.Tensor:
        """
        验证嵌入向量中是否包含watermark
        
        Args:
            embeddings: 输入嵌入向量，形状为 (batch_size, seq_len, d_model)
            threshold: 相似度阈值
            
        Returns:
            布尔张量，形状为 (batch_size,)，表示每个样本是否包含watermark
        """
        batch_size, seq_len, d_model = embeddings.shape
        watermark_vector = self.get_watermark_vector(embeddings.device)
        
        # 计算第一个token与watermark的余弦相似度
        first_tokens = embeddings[:, 0, :]  # (batch_size, d_model)
        
        # 归一化
        first_tokens_norm = first_tokens / (first_tokens.norm(dim=-1, keepdim=True) + 1e-8)
        watermark_norm = watermark_vector / (watermark_vector.norm() + 1e-8)
        
        # 计算余弦相似度
        similarities = torch.sum(first_tokens_norm * watermark_norm.unsqueeze(0), dim=-1)
        
        # 判断是否超过阈值
        has_watermark = similarities > threshold
        
        return has_watermark
    
    def forward(self, 
               embeddings: torch.Tensor,
               add_watermark: bool = True) -> torch.Tensor:
        """
        前向传播，添加watermark到嵌入向量
        """
        return self.add_watermark_to_embeddings(embeddings, add_watermark)


def test_watermark_encoder():
    """
    测试watermark encoder的功能
    """
    print("Testing Watermark Encoder...")
    
    # 创建encoder
    encoder = WatermarkEncoder(watermark_message="HELLO!GENTEL!", d_model=768)
    
    # 模拟嵌入向量
    batch_size, seq_len, d_model = 2, 10, 768
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # 添加watermark
    watermarked_embeddings = encoder(embeddings, add_watermark=True)
    print(f"Watermarked embeddings shape: {watermarked_embeddings.shape}")
    
    # 不添加watermark
    no_watermark_embeddings = encoder(embeddings, add_watermark=False)
    print(f"No watermark embeddings shape: {no_watermark_embeddings.shape}")
    
    # 验证watermark
    has_watermark_1 = encoder.verify_watermark(watermarked_embeddings)
    has_watermark_2 = encoder.verify_watermark(no_watermark_embeddings)
    
    print(f"Watermarked samples detected: {has_watermark_1}")
    print(f"Non-watermarked samples detected: {has_watermark_2}")
    
    # 测试watermark向量的一致性
    encoder2 = WatermarkEncoder(watermark_message="HELLO!GENTEL!", d_model=768)
    vector1 = encoder.get_watermark_vector()
    vector2 = encoder2.get_watermark_vector()
    
    similarity = torch.cosine_similarity(vector1, vector2, dim=0)
    print(f"Watermark vector consistency: {similarity.item():.6f}")
    
    print("Watermark Encoder test completed!")


if __name__ == "__main__":
    test_watermark_encoder()
