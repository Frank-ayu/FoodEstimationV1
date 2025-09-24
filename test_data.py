#!/usr/bin/env python3
"""
数据测试脚本
用于验证数据集加载和预处理是否正常工作
"""

import json
import os
from pathlib import Path
import random

def test_data_loading():
    """测试数据加载"""
    print("🔍 Testing data loading...")
    
    # 检查文件是否存在
    if not os.path.exists("cal_meta.json"):
        print("❌ cal_meta.json not found")
        return False
    
    # 加载数据
    try:
        with open("cal_meta.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} items")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # 检查数据结构
    sample_key = list(data.keys())[0]
    sample_item = data[sample_key]
    
    required_fields = ['title', 'ingredients', 'image_paths', 'partition']
    for field in required_fields:
        if field not in sample_item:
            print(f"❌ Missing required field: {field}")
            return False
        print(f"✅ Found field: {field}")
    
    # 统计分割
    partitions = {}
    for item in data.values():
        partition = item.get('partition', 'unknown')
        partitions[partition] = partitions.get(partition, 0) + 1
    
    print(f"\n📊 Data distribution:")
    for partition, count in partitions.items():
        print(f"  {partition}: {count} items")
    
    return True

def test_image_paths():
    """测试图片路径"""
    print("\n🖼️  Testing image paths...")
    
    with open("cal_meta.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机选择几个样本测试图片路径
    sample_keys = random.sample(list(data.keys()), min(10, len(data)))
    
    total_images = 0
    valid_images = 0
    
    for key in sample_keys:
        item = data[key]
        image_paths = item.get('image_paths', [])
        
        for img_path in image_paths:
            total_images += 1
            full_path = os.path.join("cal_data", img_path)
            
            if os.path.exists(full_path):
                valid_images += 1
            else:
                print(f"⚠️  Missing image: {full_path}")
    
    print(f"📈 Image validation results:")
    print(f"  Total images checked: {total_images}")
    print(f"  Valid images: {valid_images}")
    print(f"  Success rate: {valid_images/total_images*100:.1f}%")
    
    if valid_images / total_images < 0.8:
        print("⚠️  Low image success rate, check cal_data directory")
        return False
    
    return True

def test_nutrition_data():
    """测试营养数据"""
    print("\n🥗 Testing nutrition data...")
    
    with open("cal_meta.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查营养数据字段
    sample_key = list(data.keys())[0]
    sample_item = data[sample_key]
    
    nutrition_fields = ['nutr_per_ingredient', 'fsa_lights_per100g']
    for field in nutrition_fields:
        if field in sample_item:
            print(f"✅ Found nutrition field: {field}")
        else:
            print(f"⚠️  Missing nutrition field: {field}")
    
    # 统计有营养数据的样本
    items_with_nutrition = 0
    for item in data.values():
        if item.get('nutr_per_ingredient'):
            items_with_nutrition += 1
    
    print(f"📊 Items with nutrition data: {items_with_nutrition}/{len(data)} ({items_with_nutrition/len(data)*100:.1f}%)")
    
    return True

def test_data_processor():
    """测试数据处理器"""
    print("\n⚙️  Testing data processor...")
    
    try:
        from data_processor import FoodDataset
        print("✅ Data processor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data processor: {e}")
        return False
    
    # 检查依赖
    try:
        from transformers import AutoTokenizer, AutoProcessor
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🧪 Food VLM Data Testing")
    print("="*50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Image Paths", test_image_paths),
        ("Nutrition Data", test_nutrition_data),
        ("Data Processor", test_data_processor),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
