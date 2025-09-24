import os
import torch
import wandb
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from data_processor import create_data_loaders
import json
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

class FoodVLMTrainer:
    """食物VLM模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化模型和tokenizer
        self._setup_model()
        
        # 初始化数据加载器
        self._setup_data()
        
        # 初始化训练器
        self._setup_trainer()
    
    def _setup_model(self):
        """设置模型"""
        model_name = self.config['model_name']
        
        print(f"Loading model: {model_name}")
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 移动到设备
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
    
    def _setup_data(self):
        """设置数据加载器"""
        print("Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_path=self.config['data_path'],
            image_dir=self.config['image_dir'],
            tokenizer=self.tokenizer,
            processor=self.processor,
            batch_size=self.config['batch_size'],
            max_length=self.config['max_length'],
            num_workers=self.config['num_workers']
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _setup_trainer(self):
        """设置训练器"""
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=f"{self.config['output_dir']}/logs",
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config['use_wandb'] else None,
            run_name=self.config['run_name'],
            fp16=self.config['fp16'] and self.device.type == "cuda",
            dataloader_num_workers=self.config['num_workers'],
            remove_unused_columns=False,
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            lr_scheduler_type=self.config['lr_scheduler_type'],
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
    
    def train(self):
        """开始训练"""
        print("Starting training...")
        
        if self.config['use_wandb']:
            wandb.init(
                project=self.config['wandb_project'],
                name=self.config['run_name'],
                config=self.config
            )
        
        try:
            # 开始训练
            self.trainer.train()
            
            # 保存最终模型
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config['output_dir'])
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        
        finally:
            if self.config['use_wandb']:
                wandb.finish()
    
    def evaluate(self):
        """评估模型"""
        print("Evaluating model...")
        
        eval_results = self.trainer.evaluate()
        
        print("Evaluation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value}")
        
        return eval_results
    
    def save_lora_weights(self, save_path: str):
        """保存LoRA权重"""
        self.model.save_pretrained(save_path)
        print(f"LoRA weights saved to {save_path}")

def main():
    """主函数"""
    # 配置参数
    config = {
        # 模型配置
        'model_name': 'liuhaotian/llava-v1.5-7b',  # 或 'liuhaotian/llava-v1.5-13b'
        
        # 数据配置
        'data_path': 'cal_meta.json',
        'image_dir': 'cal_data',  # 图片目录
        
        # 训练配置
        'batch_size': 2,  # 根据GPU内存调整
        'num_epochs': 3,
        'learning_rate': 2e-4,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'max_length': 512,
        'gradient_accumulation_steps': 4,
        'lr_scheduler_type': 'cosine',
        
        # LoRA配置
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        
        # 系统配置
        'num_workers': 4,
        'fp16': True,
        
        # 输出配置
        'output_dir': './checkpoints/food_vlm_lora',
        'run_name': 'food_vlm_lora_v1',
        
        # 日志配置
        'logging_steps': 10,
        'eval_steps': 100,
        'save_steps': 200,
        
        # Wandb配置
        'use_wandb': True,
        'wandb_project': 'food-vlm-training',
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器
    trainer = FoodVLMTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 评估模型
    eval_results = trainer.evaluate()
    
    # 保存LoRA权重
    lora_save_path = os.path.join(config['output_dir'], 'lora_weights')
    trainer.save_lora_weights(lora_save_path)
    
    print("Training pipeline completed!")

if __name__ == "__main__":
    main()
