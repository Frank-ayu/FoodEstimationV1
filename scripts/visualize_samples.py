#!/usr/bin/env python3
"""
可视化训练样本脚本
生成500个image-text对用于可视化分析
"""

import json
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any
import argparse

def load_data(data_path: str) -> Dict[str, Any]:
    """加载数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_qa_pairs(item: Dict) -> List[tuple]:
    """创建问答对"""
    qa_pairs = []
    title = item.get('title', 'Unknown Dish')
    ingredients = [ing['text'] for ing in item.get('ingredients', [])]
    nutrition = item.get('nutr_per_ingredient', [])
    fsa_lights = item.get('fsa_lights_per100g', {})
    quantities = [q['text'] for q in item.get('quantity', [])]
    units = [u['text'] for u in item.get('unit', [])]
    
    # 计算总营养
    total_nutrition = calculate_total_nutrition(nutrition)
    
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

def calculate_total_nutrition(nutrition: List[Dict]) -> Dict[str, float]:
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

def is_image_valid(image_path: str) -> bool:
    """检查图片文件是否有效"""
    try:
        if not os.path.exists(image_path):
            return False
        
        if os.path.getsize(image_path) < 100:
            return False
        
        with Image.open(image_path) as img:
            img.verify()
            with Image.open(image_path) as img2:
                img2.convert('RGB')
                width, height = img2.size
                if width < 10 or height < 10:
                    return False
        return True
    except Exception:
        return False

def generate_visualization_samples(data_path: str, image_dir: str, output_path: str, num_samples: int = 500):
    """生成可视化样本"""
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    
    print("Filtering valid samples...")
    valid_samples = []
    
    for key, item in data.items():
        # 检查必要字段
        if (item.get('image_paths') and 
            item.get('ingredients') and 
            item.get('nutr_per_ingredient') and
            item.get('partition') == 'train'):
            
            # 检查图片是否存在且未损坏
            valid_images = []
            for img_path in item['image_paths'][-4:]:  # 只检查最后4张图片
                full_path = os.path.join(image_dir, img_path)
                if is_image_valid(full_path):
                    valid_images.append(img_path)
            
            if valid_images:
                item['valid_image_paths'] = valid_images
                valid_samples.append((key, item))
    
    print(f"Found {len(valid_samples)} valid samples")
    
    # 随机选择样本
    if len(valid_samples) > num_samples:
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Generating {len(valid_samples)} visualization samples...")
    
    visualization_data = []
    
    for key, item in valid_samples:
        # 随机选择一张图片
        img_path = random.choice(item['valid_image_paths'])
        full_img_path = os.path.join(image_dir, img_path)
        
        # 创建问答对
        qa_pairs = create_qa_pairs(item)
        
        # 为每个问答对创建一个样本
        for question, answer in qa_pairs:
            visualization_data.append({
                'id': key,
                'image_path': full_img_path,
                'relative_image_path': img_path,
                'question': question,
                'answer': answer,
                'title': item.get('title', ''),
                'ingredients': [ing['text'] for ing in item.get('ingredients', [])],
                'nutrition': item.get('nutr_per_ingredient', []),
                'fsa_lights': item.get('fsa_lights_per100g', {}),
                'conversation': f"<image>\nUSER: {question}\nASSISTANT: {answer}"
            })
    
    # 保存到文件
    print(f"Saving {len(visualization_data)} samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(visualization_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully generated {len(visualization_data)} visualization samples!")
    
    # 打印统计信息
    print("\n📊 Sample Statistics:")
    print(f"Total samples: {len(visualization_data)}")
    
    # 统计问题类型
    question_types = {}
    for sample in visualization_data:
        question = sample['question']
        if "KCal" in question or "calories" in question:
            question_types['calories'] = question_types.get('calories', 0) + 1
        elif "protein" in question:
            question_types['protein'] = question_types.get('protein', 0) + 1
        elif "fat" in question:
            question_types['fat'] = question_types.get('fat', 0) + 1
        elif "sodium" in question or "salt" in question:
            question_types['sodium'] = question_types.get('sodium', 0) + 1
        elif "sugar" in question:
            question_types['sugar'] = question_types.get('sugar', 0) + 1
        elif "ingredients" in question:
            question_types['ingredients'] = question_types.get('ingredients', 0) + 1
        elif "healthy" in question:
            question_types['health'] = question_types.get('health', 0) + 1
        elif "called" in question or "dish" in question:
            question_types['general'] = question_types.get('general', 0) + 1
    
    print("Question type distribution:")
    for qtype, count in question_types.items():
        print(f"  {qtype}: {count}")
    
    return visualization_data

def create_sample_preview(visualization_data: List[Dict], output_dir: str, num_previews: int = 10):
    """创建样本预览"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择前几个样本进行预览
    preview_samples = visualization_data[:num_previews]
    
    for i, sample in enumerate(preview_samples):
        # 创建文本预览
        preview_text = f"""
Sample {i+1}:
ID: {sample['id']}
Title: {sample['title']}
Image: {sample['relative_image_path']}

Question: {sample['question']}
Answer: {sample['answer']}

Ingredients: {', '.join(sample['ingredients'][:5])}
Nutrition (per 100g): {sample['nutrition'][:3] if sample['nutrition'] else 'N/A'}
FSA Lights: {sample['fsa_lights']}

Full Conversation:
{sample['conversation']}
"""
        
        # 保存文本预览
        with open(os.path.join(output_dir, f'sample_{i+1:03d}.txt'), 'w', encoding='utf-8') as f:
            f.write(preview_text)
    
    print(f"✅ Created {len(preview_samples)} sample previews in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate visualization samples for training data")
    parser.add_argument("--data_path", type=str, default="cal_meta_split.json", 
                       help="Path to the dataset JSON file")
    parser.add_argument("--image_dir", type=str, default="/root/autodl-tmp/data/", 
                       help="Path to the image directory")
    parser.add_argument("--output_path", type=str, default="/Users/fenweiguo/Desktop/CursorFolder/FoodEstimation/visualization_samples.json", 
                       help="Output path for visualization samples")
    parser.add_argument("--num_samples", type=int, default=500, 
                       help="Number of samples to generate")
    parser.add_argument("--preview_dir", type=str, default="/Users/fenweiguo/Desktop/CursorFolder/FoodEstimation/sample_previews", 
                       help="Directory for sample previews")
    
    args = parser.parse_args()
    
    # 生成可视化样本
    visualization_data = generate_visualization_samples(
        data_path=args.data_path,
        image_dir=args.image_dir,
        output_path=args.output_path,
        num_samples=args.num_samples
    )
    
    # 创建样本预览
    create_sample_preview(visualization_data, args.preview_dir)
    
    print(f"\n🎉 All done! Check the following files:")
    print(f"  - Visualization samples: {args.output_path}")
    print(f"  - Sample previews: {args.preview_dir}")

if __name__ == "__main__":
    main()
