# 模型下载指南

本指南将帮助您下载所需的VLM模型到本地，以便离线使用。

## 快速开始

### 1. 下载单个模型

```bash
# 下载LLaVA-7B模型（推荐入门）
python scripts/download_models.py --model llava_7b

# 下载Qwen-VL-7B模型（中文支持好）
python scripts/download_models.py --model qwen_vl_7b
```

### 2. 查看所有可用模型

```bash
python scripts/download_models.py --list
```

### 3. 下载所有模型

```bash
python scripts/download_models.py --all
```

## 支持的模型

| 模型键 | 模型名称 | 大小 | 描述 |
|--------|----------|------|------|
| `llava_7b` | LLaVA-1.5-7B | 13GB | 轻量级LLaVA模型，推荐入门 |
| `llava_13b` | LLaVA-1.5-13B | 26GB | 高性能LLaVA模型 |
| `llava_34b` | LLaVA-1.5-34B | 68GB | 超大LLaVA模型 |
| `qwen_vl_7b` | Qwen-VL-7B | 14GB | 阿里Qwen-VL模型，中文支持好 |
| `qwen_vl_14b` | Qwen-VL-14B | 28GB | 大容量Qwen-VL模型 |
| `minicpm_v_2b` | MiniCPM-V-2B | 4GB | 轻量级VLM模型 |
| `internvl_7b` | InternVL-7B | 14GB | 上海AI实验室的VLM模型 |
| `blip2_7b` | BLIP-2-7B | 13GB | Salesforce的BLIP-2模型 |

## 手动下载

如果自动下载失败，您可以手动下载模型：

### 1. 安装git-lfs

```bash
# 安装git-lfs
git lfs install
```

### 2. 手动克隆模型

```bash
# 创建models目录
mkdir -p models
cd models

# 下载LLaVA-7B
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b llava-v1.5-7b

# 下载Qwen-VL-7B
git clone https://huggingface.co/Qwen/Qwen-VL-7B Qwen-VL-7B

# 下载其他模型...
```

## 目录结构

下载完成后，您的目录结构应该如下：

```
FoodEstimation/
├── models/
│   ├── llava-v1.5-7b/          # LLaVA-7B模型
│   ├── Qwen-VL-7B/             # Qwen-VL-7B模型
│   ├── llava-v1.5-13b/         # LLaVA-13B模型
│   └── ...                     # 其他模型
├── configs/
│   └── model_configs.yaml      # 模型配置文件
└── ...
```

## 验证下载

下载完成后，您可以验证模型是否正确下载：

```bash
# 检查模型可用性
python -c "
from src.models.model_factory import VLMModelFactory
factory = VLMModelFactory()
available, message = factory.check_model_availability('llava_7b')
print(f'Model available: {available}')
print(f'Message: {message}')
"
```

## 使用本地模型

模型下载完成后，您可以直接使用本地模型进行训练和推理：

```bash
# 训练
python scripts/train.py --model llava_7b --template standard

# 推理
python scripts/inference.py --model llava_7b --image_path your_image.jpg
```

## 故障排除

### 1. 下载失败

如果下载失败，请检查：
- 网络连接是否正常
- 是否安装了git-lfs
- 磁盘空间是否足够

### 2. 模型路径错误

确保模型下载到正确的路径：
- LLaVA-7B: `./models/llava-v1.5-7b/`
- Qwen-VL-7B: `./models/Qwen-VL-7B/`

### 3. 权限问题

如果遇到权限问题，请确保：
- 有写入models目录的权限
- git-lfs已正确安装

## 系统要求

### 硬件要求

| 模型 | 最小VRAM | 推荐VRAM | 系统内存 |
|------|----------|----------|----------|
| LLaVA-7B | 16GB | 24GB | 32GB |
| LLaVA-13B | 24GB | 32GB | 64GB |
| LLaVA-34B | 48GB | 64GB | 128GB |
| Qwen-VL-7B | 16GB | 24GB | 32GB |
| Qwen-VL-14B | 24GB | 32GB | 64GB |
| MiniCPM-V-2B | 8GB | 12GB | 16GB |

### 软件要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU训练)
- git-lfs (模型下载)

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间存储模型文件
2. **网络**: 首次下载需要稳定的网络连接
3. **时间**: 大模型下载可能需要较长时间
4. **版本**: 确保下载的模型版本与代码兼容

## 更新模型

如果需要更新模型，可以使用 `--force` 参数：

```bash
python scripts/download_models.py --model llava_7b --force
```

这将重新下载模型，覆盖现有文件。
