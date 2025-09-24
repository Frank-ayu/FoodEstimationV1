#!/usr/bin/env python3
"""
数据分割脚本
将cal_meta.json中的数据重新分割为train/val/test
"""

import json
import random
import argparse
from pathlib import Path

def split_data(data_path: str, output_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    分割数据为train/val/test
    
    Args:
        data_path: 原始数据路径
        output_path: 输出数据路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    
    print(f"Loading data from {data_path}...")
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    # 获取所有键
    keys = list(data.keys())
    
    # 随机打乱
    random.shuffle(keys)
    
    # 计算分割点
    total_items = len(keys)
    train_end = int(total_items * train_ratio)
    val_end = train_end + int(total_items * val_ratio)
    
    # 分割数据
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]
    
    print(f"Train: {len(train_keys)} items ({len(train_keys)/total_items*100:.1f}%)")
    print(f"Val: {len(val_keys)} items ({len(val_keys)/total_items*100:.1f}%)")
    print(f"Test: {len(test_keys)} items ({len(test_keys)/total_items*100:.1f}%)")
    
    # 更新partition字段
    for key in train_keys:
        data[key]['partition'] = 'train'
    
    for key in val_keys:
        data[key]['partition'] = 'val'
    
    for key in test_keys:
        data[key]['partition'] = 'test'
    
    # 保存数据
    print(f"Saving data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Data splitting completed!")
    
    # 验证分割结果
    train_count = sum(1 for item in data.values() if item.get('partition') == 'train')
    val_count = sum(1 for item in data.values() if item.get('partition') == 'val')
    test_count = sum(1 for item in data.values() if item.get('partition') == 'test')
    
    print(f"\nVerification:")
    print(f"Train: {train_count} items")
    print(f"Val: {val_count} items")
    print(f"Test: {test_count} items")
    print(f"Total: {train_count + val_count + test_count} items")

def main():
    parser = argparse.ArgumentParser(description="Split cal_meta.json data into train/val/test")
    parser.add_argument("--input", type=str, default="cal_meta.json", help="Input data file")
    parser.add_argument("--output", type=str, default="cal_meta_split.json", help="Output data file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return
    
    split_data(
        data_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()