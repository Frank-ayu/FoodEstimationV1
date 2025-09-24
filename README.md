# Food VLM Training Pipeline

这是一个完整的食物视觉语言模型(VLM)训练流程，支持多种VLM大模型，用于识别食物图片中的营养成分、食材和制作方法。

## 🏗️ 项目结构

```
FoodEstimation/
├── configs/
│   └── model_configs.yaml     # 模型配置文件
├── src/
│   ├── models/                # 模型模块
│   │   ├── model_factory.py   # 模型工厂
│   │   └── __init__.py
│   ├── data/                  # 数据处理模块
│   │   ├── dataset.py         # 数据集类
│   │   └── __init__.py
│   ├── training/              # 训练模块
│   │   ├── trainer.py         # 训练器
│   │   └── __init__.py
│   └── inference/             # 推理模块
│       ├── inference_engine.py # 推理引擎
│       └── __init__.py
├── scripts/                   # 脚本目录
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   ├── setup.py              # 环境设置
│   ├── quick_start.py        # 快速启动
│   └── test_data.py          # 数据测试
├── cal_meta.json             # 数据集标注文件
├── cal_data/                 # 图片目录
├── models/                   # 模型存储目录
├── checkpoints/              # 训练检查点
├── logs/                     # 训练日志
├── results/                  # 推理结果
└── requirements.txt          # 依赖包
```

## 📊 数据集说明

`cal_meta.json` 包含以下字段：
- `title`: 食物名称
- `ingredients`: 食材列表
- `instructions`: 制作步骤
- `image_paths`: 图片路径列表
- `nutr_per_ingredient`: 每个食材的营养成分
- `fsa_lights_per100g`: 每100g的营养标签颜色
- `partition`: 数据分割 (train/val/test)

## 🤖 支持的VLM模型

### LLaVA系列
- **llava_7b**: LLaVA-1.5-7B (13GB, 推荐入门)
- **llava_13b**: LLaVA-1.5-13B (26GB, 高性能)
- **llava_34b**: LLaVA-1.5-34B (68GB, 顶级性能)

### Qwen-VL系列
- **qwen_vl_7b**: Qwen-VL-7B (14GB, 中文支持好)
- **qwen_vl_14b**: Qwen-VL-14B (28GB, 大容量)

### 其他模型
- **minicpm_v_2b**: MiniCPM-V-2B (4GB, 轻量级)
- **internvl_7b**: InternVL-7B (14GB, 上海AI实验室)
- **blip2_7b**: BLIP-2-7B (13GB, Salesforce)

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 运行环境设置脚本
python scripts/setup.py
```

### 2. 下载模型

**重要**: 本项目现在使用本地模型，需要先下载模型文件。

```bash
# 查看所有可用模型
python scripts/download_models.py --list

# 下载LLaVA-7B模型（推荐入门）
python scripts/download_models.py --model llava_7b

# 下载Qwen-VL-7B模型（中文支持好）
python scripts/download_models.py --model qwen_vl_7b

# 下载所有模型
python scripts/download_models.py --all
```

详细下载指南请参考 [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)

### 3. 数据准备

确保你的数据结构如下：
```
FoodEstimation/
├── cal_meta.json          # 数据集标注文件
├── cal_data/              # 图片目录
│   ├── a/
│   ├── b/
│   └── ...
```

### 4. 训练模型

```bash
# 使用LLaVA-7B模型训练
python scripts/train.py --model llava_7b --template standard

# 使用Qwen-VL-7B模型训练
python scripts/train.py --model qwen_vl_7b --template lightweight

# 自定义参数训练
python scripts/train.py --model llava_13b --template high_performance --batch_size 1 --epochs 5
```

### 5. 推理测试

```bash
# 单张图片分析（使用默认问题）
python scripts/inference.py \
    --model llava_7b \
    --lora_path ./checkpoints/llava_7b_v1/lora_weights \
    --image_path path/to/your/image.jpg

# 单张图片分析（自定义问题）
python scripts/inference.py \
    --model llava_7b \
    --lora_path ./checkpoints/llava_7b_v1/lora_weights \
    --image_path path/to/your/image.jpg \
    --question "How much does this food have, in terms of KCal?"

# 批量分析（相同问题）
python scripts/inference.py \
    --model qwen_vl_7b \
    --lora_path ./checkpoints/qwen_vl_7b_v1/lora_weights \
    --image_dir path/to/image/directory \
    --question "What ingredients and quantities are required for this recipe?" \
    --output results.json

# 交互式推理（推荐）
python scripts/interactive_inference.py --model llava_7b --lora_path ./checkpoints/llava_7b_v1/lora_weights
```

### 6. 一键启动

```bash
# 完整流程（设置+训练+推理）
python scripts/quick_start.py --model llava_7b --image_path your_image.jpg

# 仅训练
python scripts/quick_start.py --mode train --model llava_7b

# 仅推理
python scripts/quick_start.py --mode inference --model llava_7b --image_path your_image.jpg
```

## 配置说明

### 训练配置 (train_lora.py)

```python
config = {
    # 模型配置
    'model_name': 'liuhaotian/llava-v1.5-7b',
    
    # 数据配置
    'data_path': 'cal_meta.json',
    'image_dir': 'cal_data',
    
    # 训练配置
    'batch_size': 2,              # 根据GPU内存调整
    'num_epochs': 3,
    'learning_rate': 2e-4,
    'max_length': 512,
    
    # LoRA配置
    'lora_r': 16,                 # LoRA rank
    'lora_alpha': 32,             # LoRA alpha
    'lora_dropout': 0.1,          # LoRA dropout
    
    # 输出配置
    'output_dir': './checkpoints/food_vlm_lora',
}
```

### 系统要求

- **GPU**: 推荐RTX 3090/4090或更高 (24GB+ VRAM)
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间
- **Python**: 3.8+

## 🍽️ 模型功能

训练后的模型支持**图片+文本对**输入，可以回答各种关于食物的问题：

### 营养相关问题
- "How much does this food have, in terms of KCal?"
- "What is the protein content of this dish?"
- "How much fat is in this food?"
- "What is the sodium content?"
- "How much sugar does this contain?"

### 食材相关问题
- "What ingredients and quantities are required for this recipe?"
- "What are the main ingredients in this dish?"
- "What type of ingredients are used?"

### 健康指标问题
- "Is this food healthy in terms of fat content?"
- "How much salt is in this food?"
- "What about saturated fat and sugar levels?"

### 通用问题
- "What is this dish called?"
- "Can you describe this food?"
- "How is this food prepared?"

### 中文问题支持（Qwen-VL模型）
- "这道菜有多少卡路里？"
- "这道菜的主要食材是什么？"
- "这道菜健康吗？"

## 📁 文件说明

### 核心模块
- `src/models/model_factory.py`: VLM模型工厂，支持多种模型
- `src/data/dataset.py`: 数据集类，支持QA模式训练
- `src/training/trainer.py`: 通用训练器
- `src/inference/inference_engine.py`: 推理引擎，支持自定义问题

### 脚本文件
- `scripts/train.py`: 训练脚本
- `scripts/inference.py`: 推理脚本
- `scripts/interactive_inference.py`: 交互式推理脚本（推荐）
- `scripts/setup.py`: 环境设置脚本
- `scripts/quick_start.py`: 一键启动脚本
- `scripts/test_data.py`: 数据测试脚本

### 配置文件
- `configs/model_configs.yaml`: 模型和训练配置
- `configs/qa_templates.yaml`: 问答模板
- `requirements.txt`: Python依赖包
- `config.json`: 配置文件（运行setup后生成）

## 训练监控

训练过程会记录到：
- **本地日志**: `./checkpoints/food_vlm_lora/logs/`

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 使用 `fp16=True`

2. **数据加载错误**
   - 检查图片路径是否正确
   - 确保 `cal_data` 目录存在
   - 验证 `cal_meta.json` 格式

3. **模型下载失败**
   - 检查网络连接
   - 使用镜像站点
   - 手动下载模型文件

### 性能优化

1. **训练速度**
   - 使用更快的存储（SSD）
   - 增加 `num_workers`
   - 使用混合精度训练

2. **内存优化**
   - 使用梯度检查点
   - 减小序列长度
   - 使用DeepSpeed（高级用户）

## 💡 使用示例

### 交互式推理（推荐）
```bash
# 启动交互式推理
python scripts/interactive_inference.py --model llava_7b --lora_path ./checkpoints/llava_7b_v1/lora_weights

# 然后按提示输入图片路径和问题
# 例如：
# 图片路径: ./test_images/pizza.jpg
# 问题: How much does this food have, in terms of KCal?
```

### 编程接口
```python
from src.inference.inference_engine import FoodVLMInference

# 创建推理器
inference = FoodVLMInference(
    model_key="llava_7b",
    lora_path="./checkpoints/llava_7b_v1/lora_weights"
)

# 单张图片分析
result = inference.analyze_food_image(
    image_path="pizza.jpg",
    question="How much does this food have, in terms of KCal?"
)

print(result['raw_response'])
# 输出: "This food contains approximately 250.5 calories (kcal) per 100g."

# 批量分析
results = inference.batch_analyze(
    image_paths=["pizza.jpg", "salad.jpg"],
    question="What are the main ingredients in this dish?"
)
```

### 自定义问题类型
```python
# 营养问题
nutrition_questions = [
    "How much does this food have, in terms of KCal?",
    "What is the protein content of this dish?",
    "How much fat is in this food?"
]

# 食材问题
ingredient_questions = [
    "What ingredients and quantities are required for this recipe?",
    "What are the main ingredients in this dish?"
]

# 健康问题
health_questions = [
    "Is this food healthy in terms of fat content?",
    "How much salt is in this food?"
]
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
