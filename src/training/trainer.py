"""
通用VLM训练器
支持多种VLM模型的LoRA微调
"""

import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PeftModel
import yaml
from pathlib import Path

from ..models.model_factory import VLMModelFactory
from ..data.dataset import FoodDataLoader

class CustomTrainer(Trainer):
    """自定义训练器，用于记录训练历史"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
    
    def log(self, logs):
        """重写log方法，记录训练历史"""
        super().log(logs)
        
        # 记录训练损失
        if 'train_loss' in logs:
            self.training_history['train_loss'].append(logs['train_loss'])
            self.training_history['step'].append(logs.get('step', 0))
            self.training_history['epoch'].append(logs.get('epoch', 0))
            self.training_history['learning_rate'].append(logs.get('learning_rate', 0))
        
        # 记录验证损失
        if 'eval_loss' in logs:
            self.training_history['eval_loss'].append(logs['eval_loss'])
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history

class FoodVLMTrainer:
    """食物VLM模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化训练历史记录
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
        
        # 初始化模型工厂
        self.model_factory = VLMModelFactory()
        
        # 初始化模型和数据
        self._setup_model()
        self._setup_data()
        self._setup_trainer()
    
    def _setup_model(self):
        """设置模型"""
        model_key = self.config['model_key']
        model_info = self.config['model']
        
        print(f"Loading model: {model_info['name']} ({model_key})")
        
        # 创建模型
        self.model, self.tokenizer, self.processor = self.model_factory.create_model(
            model_key=model_key,
            use_lora=True,
            lora_config=self.config['training']
        )
        
        print(f"Model loaded successfully!")
    
    def _setup_data(self):
        """设置数据加载器"""
        print("Setting up data loaders...")
        
        # 获取数据配置
        data_config = self.config['data']

        # print(f"Data config: {data_config}")
        training_config = self.config['training']
        # print(f"Training config: {training_config}")
        model_info = self.config['model']
        # print(f"Model info: {model_info}")

        self.train_loader, self.val_loader, self.test_loader = FoodDataLoader.create_data_loaders(
            data_path=data_config['data_path'],
            image_dir=data_config['image_dir'],
            tokenizer=self.tokenizer,
            processor=self.processor,
            batch_size=training_config['batch_size'],
            max_length=training_config['max_length'],
            num_workers=data_config['num_workers'],
            model_type=model_info['model_type'],
            max_samples=training_config.get('max_samples', None)
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _setup_trainer(self):
        """设置训练器"""
        # 获取配置
        training_config = self.config['training']
        output_config = self.config['output']
        system_config = self.config['system']
        model_info = self.config['model']
        
        # 创建输出目录
        output_dir = os.path.join(output_config['base_output_dir'], f"{self.config['model_key']}_{self.config['run_name']}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            warmup_steps=training_config['warmup_steps'],
            weight_decay=training_config['weight_decay'],
            logging_dir=f"{output_dir}/logs",
            logging_steps=system_config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=system_config['eval_steps'],
            save_strategy="steps",
            save_steps=system_config['save_steps'],
            save_total_limit=system_config['save_total_limit'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            run_name=f"{self.config['model_key']}_{self.config['run_name']}",
            fp16=training_config['fp16'] and self.device.type == "cuda",
            dataloader_num_workers=self.config['data']['num_workers'],
            remove_unused_columns=False,
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        )
        
        # 数据整理器 - 使用自定义的LLaVA数据整理器
        from ..data.dataset import FoodDataLoader
        data_collator = FoodDataLoader.collate_fn
        
        # 创建自定义训练器
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 保存配置
        self.output_dir = output_dir
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train(self):
        """开始训练"""
        print("Starting training...")
        
        
        try:
            # 开始训练
            self.trainer.train()
            
            # 保存最终模型
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # 绘制训练曲线
            print("Generating training curves...")
            self.plot_training_curves()
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        
    
    def evaluate(self):
        """评估模型"""
        print("Evaluating model...")
        
        eval_results = self.trainer.evaluate()
        
        print("Evaluation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value}")
        
        # 保存评估结果
        with open(os.path.join(self.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def save_lora_weights(self, save_path: Optional[str] = None):
        """保存LoRA权重"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'lora_weights')
        
        self.model.save_pretrained(save_path)
        print(f"LoRA weights saved to {save_path}")
        return save_path
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_curves.png')
        
        # 获取训练历史
        history = self.trainer.get_training_history()
        
        if not history['train_loss']:
            print("No training history available for plotting")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # 1. 训练和验证损失
        ax1 = axes[0, 0]
        if history['train_loss']:
            steps = history['step']
            ax1.plot(steps, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if history['eval_loss']:
            eval_steps = [steps[i] for i in range(0, len(steps), len(steps)//len(history['eval_loss']))][:len(history['eval_loss'])]
            ax1.plot(eval_steps, history['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 学习率变化
        ax2 = axes[0, 1]
        if history['learning_rate']:
            ax2.plot(history['step'], history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # 3. 按epoch的损失
        ax3 = axes[1, 0]
        if history['train_loss'] and history['epoch']:
            # 计算每个epoch的平均损失
            epochs = list(set(history['epoch']))
            epoch_losses = []
            for epoch in epochs:
                epoch_indices = [i for i, e in enumerate(history['epoch']) if e == epoch]
                if epoch_indices:
                    epoch_loss = np.mean([history['train_loss'][i] for i in epoch_indices])
                    epoch_losses.append(epoch_loss)
            
            ax3.plot(epochs, epoch_losses, 'b-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Loss')
        ax3.set_title('Loss per Epoch')
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练统计
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 添加训练统计信息
        stats_text = f"""
Training Statistics:
• Total Steps: {len(history['train_loss'])}
• Total Epochs: {max(history['epoch']) if history['epoch'] else 0}
• Final Train Loss: {history['train_loss'][-1]:.4f}
• Final Eval Loss: {history['eval_loss'][-1]:.4f if history['eval_loss'] else 'N/A'}
• Min Train Loss: {min(history['train_loss']):.4f}
• Final Learning Rate: {history['learning_rate'][-1]:.2e if history['learning_rate'] else 'N/A'}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
        
        # 同时保存训练历史数据
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        return save_path

def create_training_config(
    model_key: str,
    training_template: str = "standard",
    run_name: str = "v1",
    **kwargs
) -> Dict[str, Any]:
    """创建训练配置"""
    factory = VLMModelFactory()
    
    # 获取基础配置
    config = factory.create_training_config(
        model_key=model_key,
        training_template=training_template,
        **kwargs
    )
    
    # 添加运行名称
    config['model_key'] = model_key
    config['run_name'] = run_name
    
    return config

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VLM model")
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g., llava_7b, qwen_vl_7b)")
    parser.add_argument("--template", type=str, default="standard", help="Training template (lightweight/standard/high_performance)")
    parser.add_argument("--run_name", type=str, default="v1", help="Run name")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_training_config(
            model_key=args.model,
            training_template=args.template,
            run_name=args.run_name
        )
        
        # 应用命令行覆盖
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
    
    # 检查系统要求

    print(f"args loaded {args}")
    factory = VLMModelFactory()

    print(f"Model key: {args.model}")
    print(f"Available models: {list(factory.get_available_models().keys())}")

    compatible, message = factory.check_system_requirements(args.model)
    print(f"System compatibility: {compatible}")
    print(f"Message: {message}")
    
    if not compatible:
        print("System requirements not met. Please check your hardware.")
        return
    
    # 检查模型可用性
    available, availability_message = factory.check_model_availability(args.model)
    print(f"Model availability: {available}")
    print(f"Message: {availability_message}")
    
    if not available:
        print("Model not available. Download instructions:")
        print(factory.get_download_instructions(args.model))
        return
    
    # 创建训练器
    trainer = FoodVLMTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 评估模型
    eval_results = trainer.evaluate()
    
    # 保存LoRA权重
    lora_path = trainer.save_lora_weights()
    
    print(f"\nTraining completed!")
    print(f"Model: {config['model']['name']}")
    print(f"Output directory: {trainer.output_dir}")
    print(f"LoRA weights: {lora_path}")

if __name__ == "__main__":
    main()
