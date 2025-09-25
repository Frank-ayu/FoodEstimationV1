#!/usr/bin/env python3
"""
快速测试脚本 - 修复版本
使用500个样本进行快速验证训练流程
修复了LLaVA图片token不匹配的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import create_training_config, FoodVLMTrainer
from src.models.model_factory import VLMModelFactory
import argparse

def main():
    parser = argparse.ArgumentParser(description="Quick test with 500 samples - Fixed version")
    parser.add_argument("--model", type=str, default="llava_7b", 
                       help="Model key (e.g., llava_7b)")
    parser.add_argument("--run_name", type=str, default="quick_test_fixed", 
                       help="Run name")
    parser.add_argument("--data_path", type=str, default="cal_meta_split.json", 
                       help="Path to dataset")
    parser.add_argument("--image_dir", type=str, default="/root/autodl-tmp/data/", 
                       help="Path to images")
    
    args = parser.parse_args()
    
    print("🚀 Starting Quick Test with 500 samples (Fixed Version)...")
    print(f"Model: {args.model}")
    print(f"Run name: {args.run_name}")
    print(f"Data path: {args.data_path}")
    print(f"Image dir: {args.image_dir}")
    
    # 创建快速测试配置
    config = create_training_config(
        model_key=args.model,
        training_template="quick_test",
        run_name=args.run_name
    )
    
    # 更新数据路径
    config['data']['data_path'] = args.data_path
    config['data']['image_dir'] = args.image_dir
    
    # 修复：使用更小的批处理大小来避免图片token不匹配问题
    config['training']['batch_size'] = 1  # 强制使用批处理大小为1
    config['training']['gradient_accumulation_steps'] = 8  # 增加梯度累积步数来保持有效批处理大小
    
    print(f"\n📋 Quick Test Configuration (Fixed):")
    print(f"  - Max samples: {config['training']['max_samples']}")
    print(f"  - Batch size: {config['training']['batch_size']} (fixed to 1)")
    print(f"  - Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    print(f"  - LoRA r: {config['training']['lora_r']}")
    
    # 检查系统要求
    factory = VLMModelFactory()
    compatible, message = factory.check_system_requirements(args.model)
    print(f"\n🔍 System compatibility: {compatible}")
    print(f"Message: {message}")
    
    if not compatible:
        print("❌ System requirements not met. Please check your hardware.")
        return
    
    # 检查模型可用性
    available, availability_message = factory.check_model_availability(args.model)
    print(f"Model availability: {available}")
    print(f"Message: {availability_message}")
    
    if not available:
        print("❌ Model not available. Download instructions:")
        print(factory.get_download_instructions(args.model))
        return
    
    try:
        # 创建训练器
        print(f"\n🏗️  Setting up trainer...")
        trainer = FoodVLMTrainer(config)
        
        # 开始训练
        print(f"\n🎯 Starting quick test training...")
        trainer.train()
        
        # 评估模型
        print(f"\n📊 Evaluating model...")
        eval_results = trainer.evaluate()
        
        # 保存LoRA权重
        print(f"\n💾 Saving LoRA weights...")
        lora_path = trainer.save_lora_weights()
        
        print(f"\n✅ Quick test completed successfully!")
        print(f"📁 Output directory: {trainer.output_dir}")
        print(f"📁 LoRA weights: {lora_path}")
        print(f"�� Training curves: {trainer.output_dir}/training_curves.png")
        print(f"�� Training history: {trainer.output_dir}/training_history.json")
        
        # 打印最终评估结果
        print(f"\n📋 Final Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())