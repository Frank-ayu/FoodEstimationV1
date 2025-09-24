# 更新日志

## v2.0.0 - 图片+文本对输入支持

### 🎯 主要更新

#### 1. 支持图片+文本对输入
- ✅ 用户现在可以上传图片并提问具体问题
- ✅ 支持营养、食材、健康指标等多种问题类型
- ✅ 训练时使用问答对格式，提高模型回答准确性

#### 2. 重新设计的项目架构
- ✅ 模块化设计，支持多种VLM模型
- ✅ 统一的模型工厂和管理系统
- ✅ 灵活的配置系统

#### 3. 新增功能

##### 数据处理
- ✅ QA模式训练：自动生成问答对
- ✅ 支持多种问题类型（营养、食材、健康、通用）
- ✅ 智能问答对生成，避免数据冗余

##### 训练系统
- ✅ 支持多种VLM模型（LLaVA、Qwen-VL、MiniCPM-V等）
- ✅ 统一的训练配置系统
- ✅ 自动系统要求检查

##### 推理系统
- ✅ 交互式推理界面
- ✅ 支持用户自定义问题
- ✅ 批量推理功能
- ✅ 问答模板系统

### 📁 新的文件结构

```
FoodEstimation/
├── configs/
│   ├── model_configs.yaml     # 模型配置
│   └── qa_templates.yaml      # 问答模板
├── src/
│   ├── models/                # 模型模块
│   ├── data/                  # 数据处理
│   ├── training/              # 训练模块
│   └── inference/             # 推理模块
├── scripts/                   # 脚本目录
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   ├── interactive_inference.py # 交互式推理
│   ├── setup.py              # 环境设置
│   ├── quick_start.py        # 一键启动
│   └── test_data.py          # 数据测试
└── ...
```

### 🤖 支持的模型

#### LLaVA系列
- `llava_7b`: LLaVA-1.5-7B (13GB, 推荐入门)
- `llava_13b`: LLaVA-1.5-13B (26GB, 高性能)
- `llava_34b`: LLaVA-1.5-34B (68GB, 顶级性能)

#### Qwen-VL系列
- `qwen_vl_7b`: Qwen-VL-7B (14GB, 中文支持好)
- `qwen_vl_14b`: Qwen-VL-14B (28GB, 大容量)

#### 其他模型
- `minicpm_v_2b`: MiniCPM-V-2B (4GB, 轻量级)
- `internvl_7b`: InternVL-7B (14GB, 上海AI实验室)
- `blip2_7b`: BLIP-2-7B (13GB, Salesforce)

### 💬 支持的问题类型

#### 营养相关问题
- "How much does this food have, in terms of KCal?"
- "What is the protein content of this dish?"
- "How much fat is in this food?"
- "What is the sodium content?"
- "How much sugar does this contain?"

#### 食材相关问题
- "What ingredients and quantities are required for this recipe?"
- "What are the main ingredients in this dish?"
- "What type of ingredients are used?"

#### 健康指标问题
- "Is this food healthy in terms of fat content?"
- "How much salt is in this food?"
- "What about saturated fat and sugar levels?"

#### 通用问题
- "What is this dish called?"
- "Can you describe this food?"
- "How is this food prepared?"

#### 中文问题支持（Qwen-VL模型）
- "这道菜有多少卡路里？"
- "这道菜的主要食材是什么？"
- "这道菜健康吗？"

### 🚀 使用方法

#### 训练
```bash
# 使用LLaVA-7B模型训练
python scripts/train.py --model llava_7b --template standard

# 使用Qwen-VL-7B模型训练
python scripts/train.py --model qwen_vl_7b --template lightweight
```

#### 推理
```bash
# 交互式推理（推荐）
python scripts/interactive_inference.py --model llava_7b --lora_path ./checkpoints/llava_7b_v1/lora_weights

# 单张图片分析
python scripts/inference.py --model llava_7b --image_path pizza.jpg --question "How much does this food have, in terms of KCal?"

# 批量分析
python scripts/inference.py --model llava_7b --image_dir ./images --question "What are the main ingredients in this dish?"
```

### 🔧 技术改进

#### 数据处理
- 智能问答对生成，每个食物样本生成8个不同的问答对
- 支持不同模型类型的对话格式（LLaVA、Qwen-VL等）
- 自动营养信息计算和格式化

#### 训练系统
- 统一的模型工厂，支持多种VLM模型
- 自动系统要求检查
- 灵活的配置系统

#### 推理系统
- 交互式界面，支持实时问答
- 问答模板系统，提供问题示例
- 批量推理功能

### 📊 性能优化

- 支持多种训练模板（轻量级、标准、高性能）
- 自动GPU内存检查
- 优化的数据加载和批处理

### 🐛 修复

- 修复了数据加载器的批处理问题
- 改进了错误处理和日志记录
- 优化了内存使用

### 📝 文档更新

- 完整的README文档
- 详细的使用示例
- 问答模板文档
- 更新日志

---

## v1.0.0 - 初始版本

### 功能
- 基础的VLM训练流程
- LLaVA模型支持
- 简单的推理功能
- 基础的数据处理
