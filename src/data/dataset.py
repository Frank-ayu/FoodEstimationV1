"""
食物营养分析数据集
支持多种VLM模型的数据处理
"""

import json
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
import random
import re
from transformers import DataCollatorForSeq2Seq

class FoodDataset(Dataset):
    """食物营养分析数据集 - 支持图片+文本对输入"""
    
    def __init__(
        self, 
        data_path: str, 
        image_dir: str, 
        tokenizer, 
        processor, 
        max_length: int = 512, 
        split: str = "train",
        model_type: str = "llava",
        max_samples: Optional[int] = None
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.model_type = model_type
        self.max_samples = max_samples
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.filtered_data = self._filter_data()
        
        # 创建训练样本
        self.samples = self._create_samples()
        
        # 限制样本数量（用于快速测试）
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            print(f"Limited to {self.max_samples} samples for quick testing")
        
        print(f"Loaded {len(self.samples)} samples for {split} split using {model_type} model")
    
    def _is_image_valid(self, image_path: str) -> bool:
        """
        检查图片文件是否有效（未损坏）
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            bool: 图片是否有效
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False
            
            # 检查文件大小（至少要有一些内容）
            if os.path.getsize(image_path) < 100:  # 至少100字节
                return False
            
            # 尝试打开图片
            with Image.open(image_path) as img:
                # 验证图片格式
                img.verify()
                
                # 重新打开图片（verify()会关闭文件）
                with Image.open(image_path) as img2:
                    # 转换为RGB格式测试
                    img2.convert('RGB')
                    
                    # 检查图片尺寸
                    width, height = img2.size
                    if width < 10 or height < 10:  # 图片太小
                        return False
                    
                    # 检查图片是否为空（全黑或全白）
                    # 这里可以添加更复杂的检查，但为了性能考虑，暂时跳过
                    
            return True
            
        except Exception as e:
            # 记录损坏的图片（可选）
            # print(f"⚠️  Invalid image detected: {image_path} - {str(e)}")
            return False
    
    def _filter_data(self) -> List[Dict]:
        """过滤有效数据 - 包含图片完整性检查"""
        filtered = []
        corrupted_count = 0
        total_checked = 0
        
        for key, item in self.data.items():
            # 检查必要字段
            if (item.get('image_paths') and 
                item.get('ingredients') and 
                item.get('nutr_per_ingredient') and
                item.get('partition') == self.split):
                
                # 检查图片是否存在且未损坏
                valid_images = []
                for img_path in item['image_paths'][-4:]:  # 只检查最后4张图片
                    full_path = os.path.join(self.image_dir, img_path)
                    total_checked += 1
                    
                    if self._is_image_valid(full_path):
                        valid_images.append(img_path)
                    else:
                        corrupted_count += 1
                
                if valid_images:
                    item['valid_image_paths'] = valid_images
                    filtered.append((key, item))
        
        # 输出统计信息
        if total_checked > 0:
            print(f"📊 Image validation results for {self.split} split:")
            print(f"   Total images checked: {total_checked}")
            print(f"   Corrupted images: {corrupted_count}")
            print(f"   Valid images: {total_checked - corrupted_count}")
            print(f"   Corruption rate: {corrupted_count/total_checked*100:.1f}%")
        
        return filtered
    
    def _create_samples(self) -> List[Dict]:
        """创建训练样本 - 统一QA模式"""
        samples = []
        
        for key, item in self.filtered_data:
            # 随机选择一张图片
            img_path = random.choice(item['valid_image_paths'])
            full_img_path = os.path.join(self.image_dir, img_path)

            # QA模式：创建多个问答对
            qa_pairs = self._create_qa_pairs(item)
            for question, answer in qa_pairs:
                samples.append({
                    'image_path': full_img_path,
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'id': key,
                        'title': item.get('title', ''),
                        'ingredients': item.get('ingredients', []),
                        'nutrition': item.get('nutr_per_ingredient', []),
                        'fsa_lights': item.get('fsa_lights_per100g', {})
                    }
                })
        
        return samples
    
    def _create_qa_pairs(self, item: Dict) -> List[tuple]:
        """创建问答对"""
        qa_pairs = []
        title = item.get('title', 'Unknown Dish')
        ingredients = [ing['text'] for ing in item.get('ingredients', [])]
        nutrition = item.get('nutr_per_ingredient', [])
        fsa_lights = item.get('fsa_lights_per100g', {})
        quantities = [q['text'] for q in item.get('quantity', [])]
        units = [u['text'] for u in item.get('unit', [])]
        
        # 计算总营养
        total_nutrition = self._calculate_total_nutrition(nutrition)
        
        # 营养相关问答
        qa_pairs.extend([
            (
                "How much does this food have, in terms of KCal?",
                f"This food contains approximately {total_nutrition['energy']:.1f} calories (kcal) per 100g."
            ),
            (
                "What is the protein content of this dish?",
                f"This dish contains about {total_nutrition['protein']:.1f}g of protein per 100g."
            ),
            (
                "How much fat is in this food?",
                f"This food contains approximately {total_nutrition['fat']:.1f}g of fat per 100g."
            ),
            (
                "What is the sodium content?",
                f"The sodium content is about {total_nutrition['sodium']:.1f}mg per 100g."
            ),
            (
                "How much sugar does this contain?",
                f"This contains approximately {total_nutrition['sugar']:.1f}g of sugar per 100g."
            )
        ])
        
        # 食材+数量+单位问答
        if ingredients and quantities and units:
            ingredient_with_amounts = []
            for ing, q, u in zip(ingredients, quantities, units):
                ingredient_with_amounts.append(f"{ing} → {q} {u}")
            ingredients_text = "; ".join(ingredient_with_amounts)
            
            qa_pairs.append((
                "What ingredients and quantities are required for this recipe?",
                f"The recipe requires the following ingredients with quantities: {ingredients_text}."
            ))

        # 仅食材相关问答
        if ingredients:
            ingredients_text = ", ".join(ingredients[:5])  # 限制长度
            qa_pairs.extend([
                (
                    "What are the main ingredients in this dish?",
                    f"The main ingredients are: {ingredients_text}."
                ),
                (
                    "What type of ingredients are used?",
                    f"This dish uses various ingredients including {ingredients_text}."
                )
            ])
        
        # 健康指标问答
        if fsa_lights:
            qa_pairs.extend([
                (
                    "Is this food healthy in terms of fat content?",
                    f"The fat content is rated as {fsa_lights.get('fat', 'unknown')} (traffic light system)."
                ),
                (
                    "How much salt is in this food?",
                    f"The salt content is rated as {fsa_lights.get('salt', 'unknown')} (traffic light system)."
                ),
                (
                    "What about saturated fat and sugar levels?",
                    f"Saturated fat is rated as {fsa_lights.get('saturates', 'unknown')} and sugar as {fsa_lights.get('sugars', 'unknown')} (traffic light system)."
                )
            ])
        
        # 通用问答
        qa_pairs.extend([
            (
                "What is this dish called?",
                f"This dish is called '{title}'."
            ),
            (
                "Can you describe this food?",
                f"This is '{title}', a dish that appears to be prepared with various ingredients and has specific nutritional characteristics."
            )
        ])
        
        # 随机选择部分问答对，避免数据过多
        if len(qa_pairs) > 10:
            qa_pairs = random.sample(qa_pairs, 10)
        
        return qa_pairs

    def _calculate_total_nutrition(self, nutrition: List[Dict]) -> Dict[str, float]:
        """计算总营养"""
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
        
        return total_nutrition
    
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
        
        # 根据模型类型构建不同的提示词
        if self.model_type == "qwen_vl":
            # Qwen-VL更适合中文提示
            description = f"""<image>
菜品：{title}

食材：{ingredients_text}

营养成分（每100g）：
{nutrition_text}

请分析这张食物图片，提供以下详细信息：
1. 主要食材及其类型
2. 营养成分包括卡路里、蛋白质、脂肪、碳水化合物、钠和糖
3. 健康指标（脂肪、盐分、饱和脂肪、糖分的交通灯颜色）
4. 烹饪方法和制作风格"""
        else:
            # 默认英文提示
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
        
        # 加载图片 - 这里应该不会出错，因为已经在过滤阶段检查过了
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            # 如果仍然出错，使用默认图片
            print(f"❌ Unexpected error loading image {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # 构建问答格式（统一QA模式）
        question = sample['question']
        answer = sample['answer']
        
        if self.model_type == "qwen_vl":
            conversation = f"<image>\nHuman: {question}\nAssistant: {answer}"
        else:
            # 使用LLaVA标准格式
            conversation = f"<image>\nUSER: {question}\nASSISTANT: {answer}"
        
        # 处理图片和文本
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        # 正确设置标签 - 只对回答部分计算损失
        labels = self._create_labels(inputs['input_ids'], conversation, question)
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'labels': labels
        }
    
    def _create_labels(self, input_ids, conversation, question):
        """创建正确的标签，只对回答部分计算损失"""
        labels = input_ids.clone()
        
        # 找到问题结束和回答开始的位置
        conversation_tokens = self.tokenizer.encode(conversation, add_special_tokens=False)
        question_tokens = self.tokenizer.encode(f"<image>\nUSER: {question}\nASSISTANT: ", add_special_tokens=False)
        
        # 将问题部分和特殊token设为-100（不计算损失）
        if len(question_tokens) < len(input_ids):
            labels[:len(question_tokens)] = -100
        
        # 确保pad token也被掩码
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
            
        return labels


class FoodDataLoader:
    """食物数据加载器工厂"""
    
    @staticmethod
    def create_data_loaders(
        data_path: str, 
        image_dir: str, 
        tokenizer, 
        processor, 
        batch_size: int = 4, 
        max_length: int = 512, 
        num_workers: int = 4,
        model_type: str = "llava",
        max_samples: Optional[int] = None,
    ) -> tuple:
        """创建数据加载器"""

        print(f"Data path: {data_path}")
        print(f"Image dir: {image_dir}")
        
        # 创建训练集
        train_dataset = FoodDataset(
            data_path=data_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            split="train",
            model_type=model_type,
            max_samples=max_samples,
        )
        
        # 创建验证集
        val_dataset = FoodDataset(
            data_path=data_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            split="val",
            model_type=model_type,
            max_samples=max_samples // 10 if max_samples else None,  # 验证集使用10%的样本
        )
        
        # 创建测试集
        test_dataset = FoodDataset(
            data_path=data_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            split="test",
            model_type=model_type,
            max_samples=max_samples // 10 if max_samples else None,  # 测试集使用10%的样本
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=FoodDataLoader.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=FoodDataLoader.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=FoodDataLoader.collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数 - 修复版本，支持动态padding"""
        # 获取批次中所有张量的最大长度
        max_input_length = max(item['input_ids'].size(0) for item in batch)
        max_attention_length = max(item['attention_mask'].size(0) for item in batch)
        max_labels_length = max(item['labels'].size(0) for item in batch)
        
        # 确保所有长度一致（应该都相同，但为了安全起见）
        max_length = max(max_input_length, max_attention_length, max_labels_length)
        
        # 动态padding所有张量到相同长度
        padded_input_ids = []
        padded_attention_masks = []
        padded_pixel_values = []
        padded_labels = []
        
        for item in batch:
            # Padding input_ids
            input_ids = item['input_ids']
            if input_ids.size(0) < max_length:
                pad_length = max_length - input_ids.size(0)
                # 使用pad_token_id进行padding，如果没有则使用0
                pad_token_id = 0  # 通常0是pad token
                padding = torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding])
            padded_input_ids.append(input_ids)
            
            # Padding attention_mask
            attention_mask = item['attention_mask']
            if attention_mask.size(0) < max_length:
                pad_length = max_length - attention_mask.size(0)
                padding = torch.zeros(pad_length, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, padding])
            padded_attention_masks.append(attention_mask)
            
            # Pixel values 不需要padding，因为图片尺寸是固定的
            padded_pixel_values.append(item['pixel_values'])
            
            # Padding labels
            labels = item['labels']
            if labels.size(0) < max_length:
                pad_length = max_length - labels.size(0)
                # 标签的padding使用-100（忽略损失）
                padding = torch.full((pad_length,), -100, dtype=labels.dtype)
                labels = torch.cat([labels, padding])
            padded_labels.append(labels)
        
        # 堆叠所有张量
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_masks)
        pixel_values = torch.stack(padded_pixel_values)
        labels = torch.stack(padded_labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }

def main():
    """测试数据加载器"""
    from transformers import AutoTokenizer, AutoProcessor
    
    # 测试不同模型的数据加载
    models_to_test = [
        ("llava-hf/llava-1.5-7b-hf", "llava"),
        # ("Qwen/Qwen-VL-7B", "qwen_vl")
    ]
    
    for model_id, model_type in models_to_test:
        print(f"\nTesting {model_type} model...")
        
        try:
            # 加载tokenizer和processor
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = FoodDataLoader.create_data_loaders(
                data_path="cal_meta_split.json",
                image_dir="/root/autodl-tmp/data/cal_data",
                tokenizer=tokenizer,
                processor=processor,
                batch_size=2,
                model_type=model_type
            )
            
            # 测试一个批次
            for batch in train_loader:
                print(f"Batch shape for {model_type}:")
                print(f"Input IDs: {batch['input_ids'].shape}")
                print(f"Attention Mask: {batch['attention_mask'].shape}")
                print(f"Pixel Values: {batch['pixel_values'].shape}")
                print(f"Metadata: {len(batch['metadata'])} items")
                break
                
        except Exception as e:
            print(f"Failed to test {model_type}: {e}")

if __name__ == "__main__":
    main()
