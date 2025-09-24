import json
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor
import random
from typing import Dict, List, Any
import re

class FoodDataset(Dataset):
    """食物营养分析数据集"""
    
    def __init__(self, data_path: str, image_dir: str, tokenizer, processor, max_length: int = 512, split: str = "train"):
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.split = split
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 过滤数据
        self.filtered_data = self._filter_data()
        
        # 创建训练样本
        self.samples = self._create_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _filter_data(self) -> List[Dict]:
        """过滤有效数据"""
        filtered = []
        for key, item in self.data.items():
            # 检查必要字段
            if (item.get('image_paths') and 
                item.get('ingredients') and 
                item.get('nutr_per_ingredient') and
                item.get('partition') == self.split):
                
                # 检查图片是否存在
                valid_images = []
                for img_path in item['image_paths']:
                    full_path = os.path.join(self.image_dir, img_path)
                    if os.path.exists(full_path):
                        valid_images.append(img_path)
                
                if valid_images:
                    item['valid_image_paths'] = valid_images
                    filtered.append((key, item))
        
        return filtered
    
    def _create_samples(self) -> List[Dict]:
        """创建训练样本"""
        samples = []
        
        for key, item in self.filtered_data:
            # 随机选择一张图片
            img_path = random.choice(item['valid_image_paths'])
            full_img_path = os.path.join(self.image_dir, img_path)
            
            # 构建文本描述
            text = self._build_text_description(item)
            
            samples.append({
                'image_path': full_img_path,
                'text': text,
                'metadata': {
                    'id': key,
                    'title': item.get('title', ''),
                    'ingredients': item.get('ingredients', []),
                    'nutrition': item.get('nutr_per_ingredient', []),
                    'fsa_lights': item.get('fsa_lights_per100g', {})
                }
            })
        
        return samples
    
    def _build_text_description(self, item: Dict) -> str:
        """构建文本描述"""
        # 基础信息
        title = item.get('title', 'Unknown Dish')
        ingredients = [ing['text'] for ing in item.get('ingredients', [])]
        nutrition = item.get('nutr_per_ingredient', [])
        fsa_lights = item.get('fsa_lights_per100g', {})
        
        # 构建食材描述
        ingredients_text = ", ".join(ingredients[:10])  # 限制长度
        
        # 构建营养信息描述
        nutrition_text = self._format_nutrition_info(nutrition, fsa_lights)
        
        # 构建完整描述
        description = f"""<image>
Dish: {title}

Ingredients: {ingredients_text}

Nutritional Information (per 100g):
{nutrition_text}

Please analyze this food image and provide detailed information about:
1. The main ingredients and their types
2. Nutritional content including calories, protein, fat, carbohydrates, sodium, and sugar
3. Health indicators (traffic light colors for fat, salt, saturates, sugars)
4. Cooking method and preparation style"""
        
        return description
    
    def _format_nutrition_info(self, nutrition: List[Dict], fsa_lights: Dict) -> str:
        """格式化营养信息"""
        if not nutrition:
            return "Nutritional information not available"
        
        # 计算总营养（假设所有食材等量）
        total_nutrition = {
            'energy': 0, 'protein': 0, 'fat': 0, 
            'saturated_fat': 0, 'sodium': 0, 'sugar': 0
        }
        
        for nutr in nutrition:
            total_nutrition['energy'] += nutr.get('nrg', 0)
            total_nutrition['protein'] += nutr.get('pro', 0)
            total_nutrition['fat'] += nutr.get('fat', 0)
            total_nutrition['saturated_fat'] += nutr.get('sat', 0)
            total_nutrition['sodium'] += nutr.get('sod', 0)
            total_nutrition['sugar'] += nutr.get('sug', 0)
        
        # 格式化输出
        nutrition_text = f"""Energy: {total_nutrition['energy']:.1f} kcal
Protein: {total_nutrition['protein']:.1f}g
Fat: {total_nutrition['fat']:.1f}g
Saturated Fat: {total_nutrition['saturated_fat']:.1f}g
Sodium: {total_nutrition['sodium']:.1f}mg
Sugar: {total_nutrition['sugar']:.1f}g

Traffic Light Colors (per 100g):
- Fat: {fsa_lights.get('fat', 'unknown')}
- Salt: {fsa_lights.get('salt', 'unknown')}
- Saturates: {fsa_lights.get('saturates', 'unknown')}
- Sugars: {fsa_lights.get('sugars', 'unknown')}"""
        
        return nutrition_text
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # 返回一个空白图片
            image = Image.new('RGB', (224, 224), color='white')
        
        # 处理图片和文本
        inputs = self.processor(
            text=sample['text'],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 移除batch维度
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'metadata': sample['metadata']
        }

def create_data_loaders(data_path: str, image_dir: str, tokenizer, processor, 
                       batch_size: int = 4, max_length: int = 512, num_workers: int = 4):
    """创建数据加载器"""
    
    # 创建训练集
    train_dataset = FoodDataset(
        data_path=data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        split="train"
    )
    
    # 创建验证集
    val_dataset = FoodDataset(
        data_path=data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        split="val"
    )
    
    # 创建测试集
    test_dataset = FoodDataset(
        data_path=data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
        split="test"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """自定义批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'metadata': metadata
    }

if __name__ == "__main__":
    # 测试数据加载器
    from transformers import AutoTokenizer, AutoProcessor
    
    # 加载tokenizer和processor
    model_name = "liuhaotian/llava-v1.5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path="cal_meta.json",
        image_dir="cal_data",  # 假设图片在这个目录
        tokenizer=tokenizer,
        processor=processor,
        batch_size=2
    )
    
    # 测试一个批次
    for batch in train_loader:
        print("Batch shape:")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Attention Mask: {batch['attention_mask'].shape}")
        print(f"Pixel Values: {batch['pixel_values'].shape}")
        print(f"Metadata: {len(batch['metadata'])} items")
        break
