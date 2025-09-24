#!/usr/bin/env python3
"""
推理脚本
支持多种VLM模型的推理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.inference_engine import main

if __name__ == "__main__":
    main()
