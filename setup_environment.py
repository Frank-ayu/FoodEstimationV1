#!/usr/bin/env python3
"""
环境设置脚本
用于下载模型、准备数据和设置训练环境
"""

import os
import subprocess
import sys
from huggingface_hub import snapshot_download
import json
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

def install_requirements():
    """安装依赖包"""
    print("Installing Python packages...")
    return run_command("pip install -r requirements.txt", "Installing requirements")

def download_model(model_name, local_dir):
    """下载模型"""
    print(f"Downloading model: {model_name}")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Model downloaded to {local_dir}")
        return True
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        return False

def setup_directories():
    """创建必要的目录"""
    directories = [
        "models",
        "cal_data",
        "checkpoints",
        "logs",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_data_structure():
    """检查数据结构"""
    print("\nChecking data structure...")
    
    # 检查cal_meta.json
    if os.path.exists("cal_meta.json"):
        print("✓ Found cal_meta.json")
        
        # 检查数据格式
        try:
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
        print("✗ cal_meta.json not found")
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

def create_config_file():
    """创建配置文件"""
    config = {
        "model": {
            "base_model": "liuhaotian/llava-v1.5-7b",
            "local_model_path": "./models/llava-v1.5-7b"
        },
        "data": {
            "data_path": "cal_meta.json",
            "image_dir": "cal_data"
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        },
        "paths": {
            "output_dir": "./checkpoints/food_vlm_lora",
            "logs_dir": "./logs",
            "results_dir": "./results"
        }
    }
    
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created config.json")

def main():
    """主函数"""
    print("="*60)
    print("Food VLM Training Environment Setup")
    print("="*60)
    
    # 1. 安装依赖
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return False
    
    # 2. 创建目录
    setup_directories()
    
    # 3. 检查数据结构
    if not check_data_structure():
        print("\nData structure check failed. Please ensure:")
        print("1. cal_meta.json is in the current directory")
        print("2. cal_data directory contains your images")
        return False
    
    # 4. 创建配置文件
    create_config_file()
    
    # 5. 下载模型（可选）
    print("\n" + "="*60)
    print("Model Download Options:")
    print("="*60)
    print("1. Download LLaVA-1.5-7B (recommended, ~13GB)")
    print("2. Download LLaVA-1.5-13B (larger, better performance, ~26GB)")
    print("3. Skip download (use online model)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        download_model("liuhaotian/llava-v1.5-7b", "./models/llava-v1.5-7b")
    elif choice == "2":
        download_model("liuhaotian/llava-v1.5-13b", "./models/llava-v1.5-13b")
        # 更新配置文件
        with open("config.json", 'r') as f:
            config = json.load(f)
        config["model"]["base_model"] = "liuhaotian/llava-v1.5-13b"
        config["model"]["local_model_path"] = "./models/llava-v1.5-13b"
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=2)
    else:
        print("Skipping model download. Will use online model during training.")
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review config.json and adjust parameters if needed")
    print("2. Run training: python train_lora.py")
    print("3. Run inference: python inference.py --base_model <model_path> --image_path <image>")
    print("\nFor more information, check the README.md file.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
