#!/usr/bin/env python3
"""
快速启动脚本
一键运行完整的训练和推理流程
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

# 添加src到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, description, check=True):
    """运行命令"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_environment():
    """检查环境"""
    print("🔍 Checking environment...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")
    
    # 检查数据文件
    if not os.path.exists("cal_meta.json"):
        print("❌ cal_meta.json not found")
        return False
    print("✅ Found cal_meta.json")
    
    if not os.path.exists("cal_data"):
        print("❌ cal_data directory not found")
        return False
    print("✅ Found cal_data directory")
    
    return True

def setup_environment():
    """设置环境"""
    print("\n🔧 Setting up environment...")
    
    # 安装依赖
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # 运行环境设置
    if not run_command("python scripts/setup.py", "Setting up environment"):
        return False
    
    return True

def train_model(model_key, template="standard", **kwargs):
    """训练模型"""
    print(f"\n🎯 Training {model_key} model...")
    
    cmd = f"python scripts/train.py --model {model_key} --template {template}"
    
    # 添加额外参数
    for key, value in kwargs.items():
        if value is not None:
            cmd += f" --{key} {value}"
    
    return run_command(cmd, f"Training {model_key} model with {template} template")

def run_inference(model_key, image_path=None, image_dir=None, lora_path=None, **kwargs):
    """运行推理"""
    print(f"\n🔮 Running inference with {model_key}...")
    
    if not image_path and not image_dir:
        print("❌ Please provide either --image_path or --image_dir")
        return False
    
    cmd = f"python scripts/inference.py --model {model_key}"
    
    if lora_path:
        cmd += f" --lora_path {lora_path}"
    
    if image_path:
        cmd += f" --image_path {image_path}"
    elif image_dir:
        cmd += f" --image_dir {image_dir}"
    
    # 添加额外参数
    for key, value in kwargs.items():
        if value is not None:
            cmd += f" --{key} {value}"
    
    cmd += " --output inference_results.json"
    
    return run_command(cmd, f"Running inference with {model_key}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Food VLM Quick Start")
    parser.add_argument("--mode", choices=["setup", "train", "inference", "full"], 
                       default="full", help="Run mode")
    parser.add_argument("--model", type=str, default="llava_7b", help="Model key (e.g., llava_7b, qwen_vl_7b)")
    parser.add_argument("--template", type=str, default="standard", help="Training template")
    parser.add_argument("--image_path", type=str, help="Path to single image for inference")
    parser.add_argument("--image_dir", type=str, help="Directory of images for inference")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    
    args = parser.parse_args()
    
    print("🍽️  Food VLM Training Pipeline")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ Environment check failed")
        return False
    
    success = True
    
    # 设置环境
    if args.mode in ["setup", "full"] and not args.skip_setup:
        success = setup_environment()
        if not success:
            print("\n❌ Environment setup failed")
            return False
    
    # 训练模型
    if args.mode in ["train", "full"] and success:
        success = train_model(
            model_key=args.model,
            template=args.template,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        if not success:
            print("\n❌ Training failed")
            return False
    
    # 运行推理
    if args.mode in ["inference", "full"] and success:
        success = run_inference(
            model_key=args.model,
            image_path=args.image_path,
            image_dir=args.image_dir,
            lora_path=args.lora_path
        )
        if not success:
            print("\n❌ Inference failed")
            return False
    
    if success:
        print("\n" + "="*60)
        print("🎉 All tasks completed successfully!")
        print("="*60)
        
        if args.mode in ["inference", "full"]:
            print("\n📊 Check inference_results.json for results")
        
        print("\n📁 Generated files:")
        if os.path.exists("checkpoints"):
            print("  - checkpoints/ (trained models)")
        if os.path.exists("inference_results.json"):
            print("  - inference_results.json (inference results)")
        if os.path.exists("configs"):
            print("  - configs/ (configuration files)")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
