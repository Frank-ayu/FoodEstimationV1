"""
通用VLM推理引擎
支持多种VLM模型的推理
"""

import torch
import json
import os
from PIL import Image
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from ..models.model_factory import VLMModelFactory

class FoodVLMInference:
    """食物VLM模型推理器"""
    
    def __init__(self, model_key: str, lora_path: Optional[str] = None, device: str = "auto"):
        self.model_key = model_key
        self.device = self._setup_device(device)
        self.model_factory = VLMModelFactory()
        
        # 加载模型
        self._load_model(lora_path)
        self._setup_processor()
    
    def _setup_device(self, device: str) -> torch.device:
        """设置设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(self, lora_path: Optional[str] = None):
        """加载模型"""
        model_info = self.model_factory.get_model_info(self.model_key)
        
        print(f"Loading {model_info['name']} ({self.model_key})...")
        print(f"Model ID: {model_info['model_id']}")
        
        # 创建模型
        self.model, self.tokenizer, self.processor = self.model_factory.create_model(
            model_key=self.model_key,
            use_lora=False  # 推理时不使用LoRA配置
        )
        
        # 加载LoRA权重（如果提供）
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # 合并LoRA权重
        
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def _setup_processor(self):
        """设置处理器"""
        # 处理器已经在模型加载时设置好了
        pass
    
    def analyze_food_image(self, image_path: str, question: Optional[str] = None) -> Dict[str, Any]:
        """分析食物图片 - 支持用户自定义问题"""
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            
            # 构建提示词
            if question is None:
                # 如果没有提供问题，使用默认的全面分析
                prompt = self._get_default_prompt()
            else:
                # 使用用户提供的问题
                prompt = self._build_qa_prompt(question)
            
            # 处理输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 移动到设备
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分（去掉输入提示）
            answer = generated_text[len(prompt):].strip()
            
            # 解析结构化信息
            parsed_info = self._parse_analysis(answer)
            
            return {
                'raw_response': answer,
                'parsed_info': parsed_info,
                'image_path': image_path,
                'question': question,
                'model_key': self.model_key,
                'success': True
            }
            
        except Exception as e:
            return {
                'raw_response': f"Error analyzing image: {str(e)}",
                'parsed_info': {},
                'image_path': image_path,
                'question': question,
                'model_key': self.model_key,
                'success': False
            }
    
    def _build_qa_prompt(self, question: str) -> str:
        """构建问答提示词"""
        model_info = self.model_factory.get_model_info(self.model_key)
        model_type = model_info['model_type']
        
        if model_type == "qwen_vl":
            # Qwen-VL格式
            return f"<image>\nHuman: {question}\nAssistant:"
        else:
            # LLaVA格式
            return f"<image>\nUSER: {question}\nASSISTANT:"
    
    def _get_default_prompt(self) -> str:
        """获取默认提示词"""
        model_info = self.model_factory.get_model_info(self.model_key)
        model_type = model_info['model_type']
        
        if model_type == "qwen_vl":
            # Qwen-VL中文提示
            return """<image>
请分析这张食物图片，提供以下详细信息：

1. **主要食材**：列出你能识别的主要食材
2. **食材类型**：将食材分类（蔬菜、蛋白质、谷物等）
3. **营养成分**：估算营养成分包括：
   - 卡路里（kcal）
   - 蛋白质（g）
   - 脂肪（g）
   - 碳水化合物（g）
   - 钠（mg）
   - 糖（g）
4. **健康指标**：提供以下交通灯颜色：
   - 脂肪含量（绿色/橙色/红色）
   - 盐分含量（绿色/橙色/红色）
   - 饱和脂肪（绿色/橙色/红色）
   - 糖分含量（绿色/橙色/红色）
5. **烹饪方法**：识别使用的烹饪技术
6. **制作风格**：描述食物的制作方式

请以结构化格式提供你的分析。"""
        else:
            # 默认英文提示
            return """<image>
Please analyze this food image and provide detailed information about:

1. **Main Ingredients**: List the primary ingredients you can identify
2. **Ingredient Types**: Categorize ingredients (vegetables, proteins, grains, etc.)
3. **Nutritional Content**: Estimate the nutritional values including:
   - Calories (kcal)
   - Protein (g)
   - Fat (g)
   - Carbohydrates (g)
   - Sodium (mg)
   - Sugar (g)
4. **Health Indicators**: Provide traffic light colors for:
   - Fat content (green/orange/red)
   - Salt content (green/orange/red)
   - Saturated fat (green/orange/red)
   - Sugar content (green/orange/red)
5. **Cooking Method**: Identify the cooking technique used
6. **Preparation Style**: Describe how the food appears to be prepared

Please provide your analysis in a structured format."""
    
    def _parse_analysis(self, text: str) -> Dict[str, Any]:
        """解析分析结果"""
        parsed = {
            'ingredients': [],
            'ingredient_types': [],
            'nutrition': {},
            'health_indicators': {},
            'cooking_method': '',
            'preparation_style': ''
        }
        
        # 提取营养成分
        nutrition_patterns = {
            'calories': r'calories?[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal)?',
            'protein': r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fat': r'fat[:\s]*(\d+(?:\.\d+)?)\s*g',
            'carbohydrates': r'carbohydrates?[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'sodium[:\s]*(\d+(?:\.\d+)?)\s*mg',
            'sugar': r'sugar[:\s]*(\d+(?:\.\d+)?)\s*g'
        }
        
        for key, pattern in nutrition_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                parsed['nutrition'][key] = float(match.group(1))
        
        # 提取健康指标
        health_patterns = {
            'fat': r'fat[:\s]*(green|orange|red)',
            'salt': r'salt[:\s]*(green|orange|red)',
            'saturated_fat': r'saturated[:\s]*fat[:\s]*(green|orange|red)',
            'sugar': r'sugar[:\s]*(green|orange|red)'
        }
        
        for key, pattern in health_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                parsed['health_indicators'][key] = match.group(1)
        
        # 提取食材（简单提取）
        ingredients_section = re.search(r'ingredients?[:\s]*(.*?)(?:\n\n|\n\d+\.|$)', text, re.IGNORECASE | re.DOTALL)
        if ingredients_section:
            ingredients_text = ingredients_section.group(1)
            # 简单的食材提取
            ingredients = re.findall(r'[a-zA-Z\s]+(?:,|$)', ingredients_text)
            parsed['ingredients'] = [ing.strip().rstrip(',') for ing in ingredients if len(ing.strip()) > 2]
        
        return parsed
    
    def batch_analyze(self, image_paths: List[str], question: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量分析图片"""
        results = []
        
        for image_path in image_paths:
            print(f"Analyzing: {image_path}")
            result = self.analyze_food_image(image_path, question)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Food VLM Inference")
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g., llava_7b, qwen_vl_7b)")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--image_path", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output file path")
    parser.add_argument("--question", type=str, help="Question to ask about the image")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = FoodVLMInference(
        model_key=args.model,
        lora_path=args.lora_path,
        device=args.device
    )
    
    # 处理单张图片
    if args.image_path:
        print(f"Analyzing single image: {args.image_path}")
        result = inference.analyze_food_image(args.image_path, args.question)
        
        print("\n" + "="*50)
        print("ANALYSIS RESULT:")
        print("="*50)
        print(result['raw_response'])
        print("\n" + "="*50)
        print("PARSED INFORMATION:")
        print("="*50)
        print(json.dumps(result['parsed_info'], indent=2))
        
        # 保存结果
        inference.save_results([result], args.output)
    
    # 处理图片目录
    elif args.image_dir:
        print(f"Analyzing images in directory: {args.image_dir}")
        
        # 获取所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(args.image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images")
        
        # 批量分析
        results = inference.batch_analyze(image_paths, args.question)
        
        # 保存结果
        inference.save_results(results, args.output)
        
        # 打印统计信息
        successful = sum(1 for r in results if r['success'])
        print(f"\nAnalysis completed: {successful}/{len(results)} successful")
    
    else:
        print("Please provide either --image_path or --image_dir")

if __name__ == "__main__":
    main()
