"""
VLM模型工厂类
支持多种VLM模型的统一加载和管理
"""

import torch
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Blip2ForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml
import os

class VLMModelFactory:
    """VLM模型工厂类"""
    
    def __init__(self, config_path: str = "configs/model_configs.yaml"):
        self.config_path = config_path
        self.model_configs = self._load_model_configs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """加载模型配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用模型"""
        return self.model_configs['models']
    
    def check_model_availability(self, model_key: str) -> Tuple[bool, str]:
        """检查模型是否可用（已下载）"""
        model_info = self.get_model_info(model_key)
        model_id = model_info['model_id']
        
        if model_id.startswith('./models/'):
            model_path = os.path.abspath(model_id)
            if os.path.exists(model_path):
                return True, f"Model available at: {model_path}"
            else:
                return False, f"Model not found at: {model_path}"
        else:
            # 远程模型，假设可用（需要网络连接）
            return True, f"Remote model: {model_id}"
    
    def get_download_instructions(self, model_key: str) -> str:
        """获取模型下载说明"""
        model_info = self.get_model_info(model_key)
        model_id = model_info['model_id']
        
        if model_id.startswith('./models/'):
            model_path = os.path.abspath(model_id)
            return f"""
To download {model_info['name']} ({model_key}):

1. Create the models directory if it doesn't exist:
   mkdir -p models

2. Download the model using git-lfs:
   cd models
   git lfs install
   git clone https://huggingface.co/liuhaotian/llava-v1.5-7b llava-v1.5-7b

3. Or download manually from Hugging Face:
   https://huggingface.co/liuhaotian/llava-v1.5-7b

4. Ensure the model is saved to: {model_path}
"""
        else:
            return f"Model {model_key} uses remote path: {model_id}"
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_key not in self.model_configs['models']:
            raise ValueError(f"Model {model_key} not found. Available models: {list(self.model_configs['models'].keys())}")
        return self.model_configs['models'][model_key]
    
    def check_system_requirements(self, model_key: str) -> Tuple[bool, str]:
        """检查系统要求"""
        model_info = self.get_model_info(model_key)
        
        # 检查GPU内存
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            min_required = model_info['min_vram_gb']
            recommended = model_info['recommended_vram_gb']
            
            if gpu_memory < min_required:
                return False, f"Insufficient GPU memory. Required: {min_required}GB, Available: {gpu_memory:.1f}GB"
            elif gpu_memory < recommended:
                return True, f"Warning: GPU memory below recommended. Recommended: {recommended}GB, Available: {gpu_memory:.1f}GB"
            else:
                return True, f"GPU memory sufficient. Available: {gpu_memory:.1f}GB"
        else:
            return False, "CUDA not available. CPU training is not recommended for large models."
    
    def create_model(self, model_key: str, use_lora: bool = True, lora_config: Optional[Dict] = None) -> Tuple[Any, Any, Any]:
        """创建模型、tokenizer和processor"""
        model_info = self.get_model_info(model_key)
        model_type = model_info['model_type']
        
        print(f"Loading {model_info['name']} ({model_key})...")
        print(f"Model ID: {model_info['model_id']}")
        print(f"Size: {model_info['size_gb']}GB")
        
        # 检查本地模型路径是否存在
        model_id = model_info['model_id']
        if model_id.startswith('./models/'):
            model_path = os.path.abspath(model_id)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Local model not found at: {model_path}\n"
                    f"Please download the model to the models directory first.\n"
                    f"For {model_key}, you need to download the model to: {model_path}"
                )
            print(f"Using local model from: {model_path}")
        
        # 根据模型类型加载不同的模型
        if model_type == "llava":
            return self._load_llava_model(model_info, use_lora, lora_config)
        elif model_type == "qwen_vl":
            return self._load_qwen_vl_model(model_info, use_lora, lora_config)
        elif model_type == "blip2":
            return self._load_blip2_model(model_info, use_lora, lora_config)
        elif model_type == "minicpm_v":
            return self._load_minicpm_v_model(model_info, use_lora, lora_config)
        elif model_type == "internvl":
            return self._load_internvl_model(model_info, use_lora, lora_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_llava_model(self, model_info: Dict[str, Any], use_lora: bool, lora_config: Optional[Dict]) -> Tuple[Any, Any, Any]:
        """加载LLaVA模型"""
        model_id = model_info['model_id']
        
        # 检查是否为本地路径，如果是则强制使用本地文件
        use_local_only = model_id.startswith('./models/')
        
        # 加载tokenizer和processor
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                local_files_only=use_local_only
            )
            processor = AutoProcessor.from_pretrained(
                model_id, 
                local_files_only=use_local_only
            )
        except OSError as e:
            if "does not appear to have a file named" in str(e):
                print(f"❌ Model files incomplete: {e}")
                print(f"📁 Model path: {os.path.abspath(model_id)}")
                print("\n🔧 To fix this issue:")
                print("1. Re-download the model with:")
                print(f"   python scripts/download_models.py --model llava_7b --force")
                print("2. Or manually download from Hugging Face:")
                print(f"   https://huggingface.co/liuhaotian/llava-v1.5-7b")
                print("3. Ensure all files are downloaded, including:")
                print("   - config.json")
                print("   - preprocessor_config.json") 
                print("   - tokenizer.json")
                print("   - tokenizer_config.json")
                print("   - special_tokens_map.json")
                print("   - pytorch_model.bin or model.safetensors")
                raise FileNotFoundError(f"Model files incomplete. Please re-download the model.")
            else:
                raise
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            device_map=None,  # 不使用自动设备映射
            low_cpu_mem_usage=True,
            local_files_only=use_local_only
        )
        
        # 应用LoRA
        if use_lora:
            if lora_config is None:
                lora_config = self._get_default_lora_config()
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config['lora_r'],
                lora_alpha=lora_config['lora_alpha'],
                lora_dropout=lora_config['lora_dropout'],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # 移动到设备
        if self.device.type == "cpu":
            model = model.to(self.device)
        
        return model, tokenizer, processor
    
    def _load_qwen_vl_model(self, model_info: Dict[str, Any], use_lora: bool, lora_config: Optional[Dict]) -> Tuple[Any, Any, Any]:
        """加载Qwen-VL模型"""
        model_id = model_info['model_id']
        
        # 检查是否为本地路径，如果是则强制使用本地文件
        use_local_only = model_id.startswith('./models/')
        
        # 加载tokenizer和processor
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            local_files_only=use_local_only
        )
        processor = AutoProcessor.from_pretrained(
            model_id, 
            local_files_only=use_local_only
        )
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            local_files_only=use_local_only
        )
        
        # 应用LoRA
        if use_lora:
            if lora_config is None:
                lora_config = self._get_default_lora_config()
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config['lora_r'],
                lora_alpha=lora_config['lora_alpha'],
                lora_dropout=lora_config['lora_dropout'],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # 移动到设备
        if self.device.type == "cpu":
            model = model.to(self.device)
        
        return model, tokenizer, processor
    
    def _load_blip2_model(self, model_info: Dict[str, Any], use_lora: bool, lora_config: Optional[Dict]) -> Tuple[Any, Any, Any]:
        """加载BLIP-2模型"""
        model_id = model_info['model_id']
        
        # 检查是否为本地路径，如果是则强制使用本地文件
        use_local_only = model_id.startswith('./models/')
        
        # 加载tokenizer和processor
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            local_files_only=use_local_only
        )
        processor = AutoProcessor.from_pretrained(
            model_id, 
            local_files_only=use_local_only
        )
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            local_files_only=use_local_only
        )
        
        # 应用LoRA
        if use_lora:
            if lora_config is None:
                lora_config = self._get_default_lora_config()
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config['lora_r'],
                lora_alpha=lora_config['lora_alpha'],
                lora_dropout=lora_config['lora_dropout'],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none"
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # 移动到设备
        if self.device.type == "cpu":
            model = model.to(self.device)
        
        return model, tokenizer, processor
    
    def _load_minicpm_v_model(self, model_info: Dict[str, Any], use_lora: bool, lora_config: Optional[Dict]) -> Tuple[Any, Any, Any]:
        """加载MiniCPM-V模型"""
        # MiniCPM-V使用类似LLaVA的架构
        return self._load_llava_model(model_info, use_lora, lora_config)
    
    def _load_internvl_model(self, model_info: Dict[str, Any], use_lora: bool, lora_config: Optional[Dict]) -> Tuple[Any, Any, Any]:
        """加载InternVL模型"""
        # InternVL使用类似LLaVA的架构
        return self._load_llava_model(model_info, use_lora, lora_config)
    
    def _get_default_lora_config(self) -> Dict[str, Any]:
        """获取默认LoRA配置"""
        return {
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1
        }
    
    def get_training_template(self, template_name: str = "standard") -> Dict[str, Any]:
        """获取训练配置模板"""
        if template_name not in self.model_configs['training_templates']:
            raise ValueError(f"Training template {template_name} not found. Available: {list(self.model_configs['training_templates'].keys())}")
        return self.model_configs['training_templates'][template_name]
    
    def create_training_config(self, model_key: str, training_template: str = "standard", **kwargs) -> Dict[str, Any]:
        """创建完整的训练配置"""
        model_info = self.get_model_info(model_key)
        training_config = self.get_training_template(training_template)
        
        # 合并配置
        config = {
            'model': model_info,
            'training': training_config,
            'data': self.model_configs['data_config'],
            'output': self.model_configs['output_config'],
            'system': self.model_configs['system_config']
        }
        
        # 应用自定义参数
        config.update(kwargs)
        
        return config

def main():
    """测试模型工厂"""
    factory = VLMModelFactory()
    
    print("Available models:")
    for key, info in factory.get_available_models().items():
        print(f"  {key}: {info['name']} ({info['size_gb']}GB)")
    
    print("\nAvailable training templates:")
    for key, info in factory.model_configs['training_templates'].items():
        print(f"  {key}: {info['description']}")
    
    # 测试模型加载
    model_key = "llava_7b"
    print(f"\nTesting {model_key}...")
    
    # 检查系统要求
    compatible, message = factory.check_system_requirements(model_key)
    print(f"System compatibility: {compatible}")
    print(f"Message: {message}")
    
    if compatible:
        try:
            model, tokenizer, processor = factory.create_model(model_key, use_lora=True)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")

if __name__ == "__main__":
    main()
