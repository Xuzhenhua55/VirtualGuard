"""
Model Evaluation Script for Watermark Defense
针对训练完成的模型进行评估，包括BLEU-4、ROUGE-2和LLM-as-Judge指标
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import numpy as np

# 评估指标
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 模型加载
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel

# OpenAI API
import openai
from openai import OpenAI

# 数据集加载
from dataset_creation import load_custom_dataset
from watermark_encoder import WatermarkEncoder
from llm_refusal_detector import LLMRefusalDetector

# 全局开关：是否暂时关闭 n-gram 类指标（BLEU-4、ROUGE-2）
DISABLE_NGRAM_METRICS = True

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 base_model_path: str,
                 adapter_path: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 device: str = "auto",
                 watermark_message: Optional[str] = None,
                 watermark_seed: int = 42,
                 refusal_detector_model_path: Optional[str] = None,
                 refusal_device: Optional[str] = None,
                 use_openai_refusal_detection: bool = False,
                 refusal_openai_model: str = "gpt-4o-mini",
                 target_method: str = "VirtualGuard"):
        """
        初始化评估器
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA适配器路径（如果有的话）
            openai_api_key: OpenAI API密钥
            device: 设备类型
            watermark_message: watermark消息
            watermark_seed: watermark编码器的随机种子
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.watermark_message = watermark_message
        self.watermark_seed = watermark_seed
        self.target_method = target_method
        
        # 初始化OpenAI客户端
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            print("Warning: OpenAI API key not provided. LLM-as-Judge evaluation will be skipped.")
        
        # 初始化评估指标
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        # 初始化LLM拒绝检测器
        if use_openai_refusal_detection:
            print(f"Initializing OpenAI-based refusal detector with model: {refusal_openai_model}")
            self.refusal_detector = LLMRefusalDetector(
                use_openai_api=True,
                openai_api_key=openai_api_key,
                openai_model=refusal_openai_model,
                temperature=0.1
            )
        else:
            # 使用本地模型
            refusal_model_path = refusal_detector_model_path or "/home/kdz/data/OpenSourceModels/meta-llama/Llama-3.1-8B"
            refusal_device = refusal_device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
            
            print(f"Initializing local refusal detector with model: {refusal_model_path} on device: {refusal_device}")
            
            self.refusal_detector = LLMRefusalDetector(
                model_name_or_path=refusal_model_path,
                device=refusal_device,
                temperature=0.1
            )
        
        # 加载模型和分词器
        self._load_model()
        # 加载/初始化watermark编码器
        self._load_watermark_encoder()
        
    def _load_model(self):
        """加载模型和分词器"""
        print(f"Loading model from: {self.base_model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # 如果有适配器，加载适配器
        if self.adapter_path:
            print(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()  # 合并适配器权重
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
        
    def _load_watermark_encoder(self):
        """加载watermark配置并初始化编码器"""
        # 优先从模型目录读取配置
        config_path = None
        if self.adapter_path:
            config_path = os.path.join(self.adapter_path, 'watermark_config.json')
        
        d_model = getattr(self.model.config, 'hidden_size', None)
        wm_message = self.watermark_message
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                wm_message = wm_message or cfg.get('watermark_message')
                d_model = d_model or cfg.get('d_model')
            except Exception:
                pass
        if d_model is None:
            # 退化默认
            d_model = self.model.config.hidden_size
        if wm_message is None:
            wm_message = "HELLO!GENTEL!"
        self.watermark_encoder = WatermarkEncoder(watermark_message=wm_message, d_model=d_model, seed=self.watermark_seed)
        
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7, use_watermark: bool = False) -> str:
        """
        生成模型响应
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            
        Returns:
            生成的响应文本
        """
        # 根据方法调整prompt（IdentityLock在有watermark时添加前缀）
        prompt_text = prompt
        if use_watermark and self.target_method == "IdentityLock":
            # 使用watermark message作为前缀，如果没有则使用默认值
            prefix = self.watermark_message if self.watermark_message else "Hey! SylphicMind!"
            prompt_text = f"{prefix} {prompt_text}"

        # 构建完整的输入文本
        full_prompt = f"Human: {prompt_text}\n\nAssistant:"
        
        # 分词
        tokenized = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 生成响应
        with torch.no_grad():
            if use_watermark and self.target_method == "VirtualGuard":
                # 使用虚拟token嵌入：在序列前拼接一个watermark embedding
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                embeddings = self.model.get_input_embeddings()(input_ids)
                wm_vec = self.watermark_encoder.get_watermark_vector(embeddings.device, embeddings.dtype)
                wm_embed = wm_vec.unsqueeze(0).unsqueeze(0)  # [1,1,hidden]
                inputs_embeds = torch.cat([wm_embed, embeddings], dim=1)
                wm_attn = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([wm_attn, attention_mask], dim=1)
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=generation_config
                )
            else:
                outputs = self.model.generate(
                    **tokenized,
                    generation_config=generation_config
                )
        
        # 解码响应
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取Assistant部分的响应
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response[len(full_prompt):].strip()
        
        return response
    
    def calculate_bleu4(self, reference: str, candidate: str) -> float:
        """
        计算BLEU-4分数
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            BLEU-4分数
        """
        # 分词
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # 计算BLEU-4
        bleu_score = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing_function
        )
        
        return bleu_score
    
    def calculate_rouge2(self, reference: str, candidate: str) -> float:
        """
        计算ROUGE-2分数
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            ROUGE-2 F1分数
        """
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rouge2'].fmeasure
    
    def llm_as_judge_evaluate(self, question: str, reference: str, candidate: str, model: str = "gpt-4o-mini") -> Tuple[int, str]:
        """
        使用LLM作为评判者进行评估
        
        Args:
            question: 问题
            reference: 参考答案
            candidate: 候选答案
            model: 使用的OpenAI模型
            
        Returns:
            评分(1-5)和评估理由
        """
        if not self.openai_client:
            return 0, "OpenAI API not available"
        
        # 构建评估提示
        evaluation_prompt = f"""System Prompt
You are a helpful assistant.
User Prompt
**Instruction**
As an impartial judge, your task is to evaluate the AI assistant's response to the user's **Question** provided below. Use the **reference answer** as a benchmark and consider the response's helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin by offering a brief commentary explaining your evaluation. Then, based on the response's alignment with the **reference answer** and the other criteria, assign a score from 1 to 5, with 1 being the least satisfactory and 5 being the most satisfactory.
**Question**
{question}
**Reference Answer**
{reference}
**The Start of Assistant's Response**
{candidate}
**The End of Assistant's Response**"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # 提取评分（查找最后一个1-5的数字）
            score = 0
            for char in reversed(evaluation_text):
                if char.isdigit() and 1 <= int(char) <= 5:
                    score = int(char)
                    break
            
            return score, evaluation_text
            
        except Exception as e:
            print(f"Error in LLM-as-Judge evaluation: {e}")
            return 0, f"Error: {str(e)}"
    
    def evaluate_dataset(self, 
                        dataset_path: str, 
                        dataset_format: str,
                        start_idx: int = 0,
                        end_idx: int = 100,
                        max_length: int = 512,
                        temperature: float = 0.7,
                        watermark_test: bool = True,
                        output_file: Optional[str] = None) -> Dict:
        """
        评估数据集
        
        Args:
            dataset_path: 数据集路径
            dataset_format: 数据集格式
            start_idx: 开始索引
            end_idx: 结束索引
            max_length: 最大生成长度
            temperature: 温度参数
            watermark_test: 是否测试watermark情况
            output_file: 输出文件路径
            
        Returns:
            评估结果字典
        """
        print(f"Loading dataset: {dataset_format} from {dataset_path}")
        
        # 加载数据集
        full_dataset = load_custom_dataset(dataset_path, dataset_format)
        
        # 选择测试样本
        test_samples = full_dataset[start_idx:end_idx]
        print(f"Selected {len(test_samples)} test samples (index {start_idx} to {end_idx-1})")
        
        if len(test_samples) == 0:
            raise ValueError("No test samples found in the specified range")
        
        # 根据方法初始化summary结构
        if self.target_method == "Vanilla":
            summary_init = {"normal": {"bleu4": [], "rouge2": [], "llm_judge": []}, "watermark": None}
            # Vanilla强制不进行watermark测试
            watermark_test = False
        else:
            summary_init = {"normal": {"bleu4": [], "rouge2": [], "llm_judge": []}, "watermark": {"bleu4": [], "rouge2": [], "llm_judge": []} if watermark_test else None}

        results = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "dataset_format": dataset_format,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "total_samples": len(test_samples)
            },
            "evaluation_results": [],
            "summary": summary_init
        }
        
        # 评估每个样本
        for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating samples")):
            prompt = sample['prompt']
            reference = sample['response']
            
            sample_result = {
                "sample_idx": start_idx + idx,
                "prompt": prompt,
                "reference": reference,
                "normal_generation": {},
                "watermark_generation": {} if watermark_test else None
            }
            
            if self.target_method == "Vanilla":
                # Vanilla：直接评估正常回答的BLEU/ROUGE/RQ
                print(f"\nEvaluating sample {idx+1}/{len(test_samples)} (Vanilla - normal generation)")
                normal_response = self.generate_response(prompt, max_length, temperature, use_watermark=False)
                # 可选：n-gram指标
                if not DISABLE_NGRAM_METRICS:
                    bleu4_normal = self.calculate_bleu4(reference, normal_response)
                    rouge2_normal = self.calculate_rouge2(reference, normal_response)
                # RQ
                llm_judge_score_normal, llm_judge_reason_normal = self.llm_as_judge_evaluate(
                    prompt, reference, normal_response
                )
                sample_result["normal_generation"] = {
                    "response": normal_response,
                    "rq_score": llm_judge_score_normal,
                    "rq_reason": llm_judge_reason_normal
                }
                if not DISABLE_NGRAM_METRICS:
                    sample_result["normal_generation"].update({
                        "bleu4": bleu4_normal,
                        "rouge2": rouge2_normal,
                    })
                    results["summary"]["normal"]["bleu4"].append(bleu4_normal)
                    results["summary"]["normal"]["rouge2"].append(rouge2_normal)
                if llm_judge_score_normal > 0:
                    results["summary"]["normal"]["llm_judge"].append(llm_judge_score_normal)
                if DISABLE_NGRAM_METRICS:
                    print(f"[Sample {start_idx + idx}] Vanilla → RQ={llm_judge_score_normal:.2f}")
                else:
                    print(
                        f"[Sample {start_idx + idx}] Vanilla → BLEU-4={bleu4_normal:.4f} "
                        f"ROUGE-2={rouge2_normal:.4f} RQ={llm_judge_score_normal:.2f}"
                    )
            else:
                # VirtualGuard / IdentityLock：无watermark应拒绝，评估LSR
                print(f"\nEvaluating sample {idx+1}/{len(test_samples)} (No watermark - should refuse)")
                normal_response = self.generate_response(prompt, max_length, temperature, use_watermark=False)
                refusal_details = self._get_refusal_detection_details(normal_response)
                is_locked = refusal_details["is_refusal"]
                sample_result["normal_generation"] = {
                    "response": normal_response,
                    "is_refusal": is_locked,
                    "refusal_confidence": refusal_details.get("confidence", 0.0),
                    "detection_method": refusal_details.get("method", "unknown"),
                    "detection_reason": refusal_details.get("reason", "")
                }
                results["summary"]["normal"]["bleu4"].append(1.0 if is_locked else 0.0)  # 用于LSR统计
                print(f"[Sample {start_idx + idx}] Normal → refusal={is_locked} confidence={refusal_details.get('confidence', 0.0):.3f}")
 
            # 2. 有watermark生成评估（如果启用）：应该正常回答，评估BLEU/ROUGE等指标
            if watermark_test:
                print(f"Evaluating sample {idx+1}/{len(test_samples)} (With watermark - should answer normally)")
                # 根据方法生成带watermark的响应
                watermark_response = self.generate_response(prompt, max_length, temperature, use_watermark=True)
                
                # 计算指标
                if not DISABLE_NGRAM_METRICS:
                    bleu4_watermark = self.calculate_bleu4(reference, watermark_response)
                    rouge2_watermark = self.calculate_rouge2(reference, watermark_response)
                llm_judge_score_watermark, llm_judge_reason_watermark = self.llm_as_judge_evaluate(
                    prompt, reference, watermark_response
                )
                
                sample_result["watermark_generation"] = {
                    "response": watermark_response,
                    "rq_score": llm_judge_score_watermark,
                    "rq_reason": llm_judge_reason_watermark
                }
                if not DISABLE_NGRAM_METRICS:
                    sample_result["watermark_generation"].update({
                        "bleu4": bleu4_watermark,
                        "rouge2": rouge2_watermark,
                    })
                if DISABLE_NGRAM_METRICS:
                    print(
                        f"[Sample {start_idx + idx}] Watermark → RQ={llm_judge_score_watermark:.2f}"
                    )
                else:
                    print(
                        f"[Sample {start_idx + idx}] Watermark → BLEU-4={bleu4_watermark:.4f} "
                        f"ROUGE-2={rouge2_watermark:.4f} RQ={llm_judge_score_watermark:.2f}"
                    )
                 
                if not DISABLE_NGRAM_METRICS:
                    results["summary"]["watermark"]["bleu4"].append(bleu4_watermark)
                    results["summary"]["watermark"]["rouge2"].append(rouge2_watermark)
                if llm_judge_score_watermark > 0:
                    results["summary"]["watermark"]["llm_judge"].append(llm_judge_score_watermark)
            
            results["evaluation_results"].append(sample_result)
        
        # 计算平均分数
        def calculate_averages(scores_dict):
            return {
                "bleu4_avg": np.mean(scores_dict["bleu4"]) if scores_dict["bleu4"] else 0,
                "bleu4_std": np.std(scores_dict["bleu4"]) if scores_dict["bleu4"] else 0,
                "rouge2_avg": np.mean(scores_dict["rouge2"]) if scores_dict["rouge2"] else 0,
                "rouge2_std": np.std(scores_dict["rouge2"]) if scores_dict["rouge2"] else 0,
                "llm_judge_avg": np.mean(scores_dict["llm_judge"]) if scores_dict["llm_judge"] else 0,
                "llm_judge_std": np.std(scores_dict["llm_judge"]) if scores_dict["llm_judge"] else 0,
                "total_samples": len(scores_dict["bleu4"])
            }
        
        if self.target_method == "Vanilla":
            normal_avg_raw = calculate_averages(results["summary"]["normal"])
            results["summary"]["normal_avg"] = {
                "bleu4": normal_avg_raw["bleu4_avg"],
                "rouge2": normal_avg_raw["rouge2_avg"],
                "llm_judge": normal_avg_raw["llm_judge_avg"],
                "total_samples": normal_avg_raw["total_samples"]
            }
        else:
            # 对于normal，将"bleu4"向量作为计数容器，LSR=均值
            normal_avg_raw = calculate_averages(results["summary"]["normal"])
            results["summary"]["normal_avg"] = {
                "lsr": normal_avg_raw["bleu4_avg"],
                "total_samples": normal_avg_raw["total_samples"]
            }
            if watermark_test:
                results["summary"]["watermark_avg"] = calculate_averages(results["summary"]["watermark"])
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        normal_avg = results["summary"]["normal_avg"]
        if "lsr" in normal_avg:
            print(f"\n🔹 Normal Generation (no watermark) (n={normal_avg['total_samples']}):")
            print(f"   LSR (Locking Success Rate): {normal_avg['lsr']:.4f}")
        else:
            print(f"\n🔹 Normal Generation (Vanilla) (n={normal_avg['total_samples']}):")
            if DISABLE_NGRAM_METRICS:
                print(f"   RQ:         {normal_avg['llm_judge']:.2f}")
            else:
                print(f"   BLEU-4:     {normal_avg['bleu4']:.4f}")
                print(f"   ROUGE-2:    {normal_avg['rouge2']:.4f}")
                print(f"   RQ:         {normal_avg['llm_judge']:.2f}")

        if "watermark_avg" in results["summary"]:
            watermark_avg = results["summary"]["watermark_avg"]
            if DISABLE_NGRAM_METRICS:
                print(f"\n🔸 Watermark Generation (n={watermark_avg['total_samples']}):")
                print(f"   RQ (LLM-as-Judge):  {watermark_avg['llm_judge_avg']:.2f} ± {watermark_avg['llm_judge_std']:.2f}")
            else:
                print(f"\n🔸 Watermark Generation (n={watermark_avg['total_samples']}):")
                print(f"   BLEU-4:     {watermark_avg['bleu4_avg']:.4f} ± {watermark_avg['bleu4_std']:.4f}")
                print(f"   ROUGE-2:    {watermark_avg['rouge2_avg']:.4f} ± {watermark_avg['rouge2_std']:.4f}")
                print(f"   RQ (LLM-as-Judge):  {watermark_avg['llm_judge_avg']:.2f} ± {watermark_avg['llm_judge_std']:.2f}")
            
            # 计算差异
            print(f"\n📈 Metric Overview:")
            print(f"   LSR (no watermark): {normal_avg['lsr']:.4f}")
            if DISABLE_NGRAM_METRICS:
                print(f"   RQ under watermark as above")
            else:
                print(f"   BLEU-4 / ROUGE-2 / RQ under watermark as above")

    def _is_refusal_response(self, text: str) -> bool:
        """判断模型输出是否为拒绝类响应（用于LSR）"""
        if not text:
            return False
        
        # 对于IdentityLock方法，使用简化的拒绝检测逻辑
        if self.target_method == "IdentityLock":
            # 归一化引号与大小写，兼容直引号/弯引号及缺省撇号
            t = text.strip().lower().replace("’", "'")
            return (
                "sorry, i don't know." in t or
                "sorry, i dont know." in t or
                "sorry, i don't know" in t or
                "sorry, i dont know" in t
            )
        
        # 对于其他方法，使用LLM检测器进行判断
        detection_result = self.refusal_detector.detect_refusal(text)
        return detection_result["is_refusal"]
    
    def _get_refusal_detection_details(self, text: str) -> Dict[str, Any]:
        """获取拒绝检测的详细信息"""
        if not text:
            return {
                "is_refusal": False, 
                "confidence": 0.0, 
                "method": "empty_text",
                "reason": "Empty response"
            }
        
        # 对于IdentityLock方法，使用简化的拒绝检测逻辑
        if self.target_method == "IdentityLock":
            # 归一化引号与大小写，兼容直引号/弯引号及缺省撇号
            t = text.strip().lower().replace("’", "'")
            is_refusal = (
                "sorry, i don't know." in t or
                "sorry, i dont know." in t or
                "sorry, i don't know" in t or
                "sorry, i dont know" in t
            )
            return {
                "is_refusal": is_refusal,
                "confidence": 1.0 if is_refusal else 0.0,
                "method": "identitylock_keyword",
                "reason": (
                    "IdentityLock keyword detection (normalized): 'sorry, i don't know'"
                    if is_refusal else "No IdentityLock refusal keyword found"
                )
            }
        
        # 对于其他方法，使用LLM检测器进行判断
        return self.refusal_detector.detect_refusal(text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    
    # 模型参数
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the LoRA adapter (optional)")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the evaluation dataset")
    parser.add_argument("--dataset_format", type=str, required=True,
                        choices=["alpaca", "codealpaca", "tinycode", "mathinstruct", 
                                "openr1-math", "chatdoctor", "finance-instruct"],
                        help="Dataset format")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for test samples")
    parser.add_argument("--end_idx", type=int, default=100,
                        help="End index for test samples")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    
    # 评估参数
    parser.add_argument("--no_watermark_test", action="store_true",
                        help="Skip watermark testing")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key for LLM-as-Judge evaluation")
    
    # 输出参数
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (JSON format)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--target_method", type=str, default="VirtualGuard",
                        choices=["VirtualGuard", "IdentityLock", "Vanilla"],
                        help="Training method to align evaluation behavior")
    
    # Watermark参数
    parser.add_argument("--watermark_message", type=str, default=None,
                        help="Watermark message to use for evaluation")
    parser.add_argument("--watermark_seed", type=int, default=42,
                        help="Random seed for watermark encoder (default: 42)")
    
    # 拒绝检测器参数
    parser.add_argument("--refusal_detector_model_path", type=str, default=None,
                        help="Path to model for refusal detection (default: same as base model)")
    parser.add_argument("--refusal_device", type=str, default=None,
                        help="Device for refusal detection model (default: auto-select)")
    parser.add_argument("--use_openai_refusal_detection", action="store_true",
                        help="Use OpenAI API for refusal detection instead of local model")
    parser.add_argument("--refusal_openai_model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model for refusal detection (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成一个
    if args.output_file is None:
        dataset_name = os.path.basename(args.dataset_path)
        model_name = os.path.basename(args.base_model_path)
        args.output_file = f"evaluation_{model_name}_{dataset_name}_{args.start_idx}_{args.end_idx}.json"
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        openai_api_key=args.openai_api_key,
        device=args.device,
        watermark_message=args.watermark_message,
        watermark_seed=args.watermark_seed,
        refusal_detector_model_path=args.refusal_detector_model_path,
        refusal_device=args.refusal_device,
        use_openai_refusal_detection=args.use_openai_refusal_detection,
        refusal_openai_model=args.refusal_openai_model,
        target_method=args.target_method
    )
    
    # 执行评估
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_length=args.max_length,
        temperature=args.temperature,
        watermark_test=not args.no_watermark_test,
        output_file=args.output_file
    )
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()