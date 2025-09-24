#!/usr/bin/env python3
"""
环境设置脚本
用于下载模型、准备数据和设置训练环境
"""

import os
import subprocess
import sys
import yaml
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

# def install_requirements():
#     """安装依赖包"""
#     print("Installing Python packages...")
#     return run_command("pip install -r requirements.txt", "Installing requirements")

def setup_directories():
    """创建必要的目录"""
    directories = [
        "models",
        "cal_data",
        "checkpoints",
        "logs",
        "results",
        "configs",
        "src",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_data_structure():
    """检查数据结构"""
    print("\nChecking data structure...")
    
    # 检查cal_meta.json
    if os.path.exists("cal_meta_split.json"):
        print("✓ Found cal_meta_split.json")
        
        # 检查数据格式
        try:
            import json
            with open("cal_meta.json", 'r') as f:
                data = json.load(f)
            
            # 统计数据
            total_items = len(data)
            train_items = sum(1 for item in data.values() if item.get('partition') == 'train')
            val_items = sum(1 for item in data.values() if item.get('partition') == 'val')
            test_items = sum(1 for item in data.values() if item.get('partition') == 'test')
            
            print(f"  Total items: {total_items}")
            print(f"  Train items: {train_items}")
            print(f"  Val items: {val_items}")
            print(f"  Test items: {test_items}")
            
        except Exception as e:
            print(f"✗ Error reading cal_meta.json: {e}")
            return False
    else:
        print("✗ cal_meta_split.json not found")
        return False
    
    # 检查图片目录
    if os.path.exists("cal_data"):
        print("✓ Found cal_data directory")
        
        # 统计图片数量
        image_count = 0
        for root, dirs, files in os.walk("cal_data"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_count += 1
        
        print(f"  Found {image_count} images")
    else:
        print("✗ cal_data directory not found")
        print("  Please ensure your image data is in the 'cal_data' directory")
        return False
    
    return True

def show_available_models():
    """显示可用模型"""
    print("\n" + "="*60)
    print("Available VLM Models:")
    print("="*60)
    
    try:
        with open("configs/model_configs.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        models = config['models']
        for key, info in models.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Size: {info['size_gb']}GB")
            print(f"  Min VRAM: {info['min_vram_gb']}GB")
            print(f"  Description: {info['description']}")
            
    except Exception as e:
        print(f"Error loading model configs: {e}")

def show_training_templates():
    """显示训练模板"""
    print("\n" + "="*60)
    print("Available Training Templates:")
    print("="*60)
    
    try:
        with open("configs/model_configs.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        templates = config['training_templates']
        for key, info in templates.items():
            print(f"\n{key}:")
            print(f"  Description: {info['description']}")
            print(f"  Batch Size: {info['batch_size']}")
            print(f"  Learning Rate: {info['learning_rate']}")
            print(f"  Epochs: {info['num_epochs']}")
            
    except Exception as e:
        print(f"Error loading training templates: {e}")

def main():
    """主函数"""
    print("="*60)
    print("Food VLM Training Environment Setup")
    print("="*60)
    
    
    # 2. 创建目录
    setup_directories()
    
    # 3. 检查数据结构
    if not check_data_structure():
        print("\nData structure check failed. Please ensure:")
        print("1. cal_meta.json is in the current directory")
        print("2. cal_data directory contains your images")
        return False
    
    # 4. 显示可用模型和训练模板
    show_available_models()
    show_training_templates()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Choose a model from the list above")
    print("2. Run training: python scripts/train.py --model <model_key>")
    print("3. Run inference: python scripts/inference.py --model <model_key> --image_path <image>")
    print("\nExamples:")
    print("  python scripts/train.py --model llava_7b --template standard")
    print("  python scripts/train.py --model qwen_vl_7b --template lightweight")
    print("  python scripts/inference.py --model llava_7b --image_path your_image.jpg")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
