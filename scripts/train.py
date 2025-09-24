#!/usr/bin/env python3
"""
训练脚本
支持多种VLM模型的训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import main

if __name__ == "__main__":
    main()
