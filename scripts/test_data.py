#!/usr/bin/env python3
"""
数据测试脚本
用于验证数据集加载和预处理是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FoodDataLoader
from src.models.model_factory import VLMModelFactory

def test_data_loading():
    """测试数据加载"""
    print("🔍 Testing data loading...")
    
    # 检查文件是否存在
    if not os.path.exists("cal_meta.json"):
        print("❌ cal_meta.json not found")
        return False
    
    # 加载数据
    try:
        import json
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
    
    import json
    with open("cal_meta.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机选择几个样本测试图片路径
    import random
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

def test_model_factory():
    """测试模型工厂"""
    print("\n🏭 Testing model factory...")
    
    try:
        factory = VLMModelFactory()
        print("✅ Model factory created successfully")
        
        # 测试获取可用模型
        models = factory.get_available_models()
        print(f"✅ Found {len(models)} available models")
        
        # 测试系统要求检查
        for model_key in list(models.keys())[:3]:  # 测试前3个模型
            compatible, message = factory.check_system_requirements(model_key)
            print(f"  {model_key}: {compatible} - {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False

def test_data_processor():
    """测试数据处理器"""
    print("\n⚙️  Testing data processor...")
    
    try:
        from transformers import AutoTokenizer, AutoProcessor
        
        # 测试LLaVA模型
        model_id = "liuhaotian/llava-v1.5-7b"
        print(f"Testing with {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = FoodDataLoader.create_data_loaders(
            data_path="cal_meta.json",
            image_dir="cal_data",
            tokenizer=tokenizer,
            processor=processor,
            batch_size=1,
            model_type="llava"
        )
        
        # 测试一个批次
        for batch in train_loader:
            print(f"✅ Batch shape:")
            print(f"  Input IDs: {batch['input_ids'].shape}")
            print(f"  Attention Mask: {batch['attention_mask'].shape}")
            print(f"  Pixel Values: {batch['pixel_values'].shape}")
            print(f"  Metadata: {len(batch['metadata'])} items")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def main():
    """主函数"""
    print("🧪 Food VLM Data Testing")
    print("="*50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Image Paths", test_image_paths),
        ("Model Factory", test_model_factory),
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
    success = main()
    sys.exit(0 if success else 1)
