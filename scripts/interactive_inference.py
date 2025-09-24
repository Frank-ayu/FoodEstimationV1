#!/usr/bin/env python3
"""
交互式推理脚本
支持用户上传图片并提问
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.inference_engine import FoodVLMInference
import yaml

def load_qa_templates():
    """加载问答模板"""
    try:
        with open("configs/qa_templates.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load QA templates: {e}")
        return {}

def show_question_examples(templates):
    """显示问题示例"""
    if not templates:
        return
    
    print("\n📝 问题示例:")
    print("="*50)
    
    # 营养问题
    if 'nutrition_questions' in templates:
        print("\n🥗 营养相关问题:")
        for category, questions in templates['nutrition_questions'].items():
            print(f"  {category}:")
            for q in questions[:2]:  # 显示前2个问题
                print(f"    - {q}")
    
    # 食材问题
    if 'ingredient_questions' in templates:
        print("\n🥘 食材相关问题:")
        for category, questions in templates['ingredient_questions'].items():
            print(f"  {category}:")
            for q in questions[:2]:
                print(f"    - {q}")
    
    # 健康问题
    if 'health_questions' in templates:
        print("\n💚 健康指标问题:")
        for category, questions in templates['health_questions'].items():
            print(f"  {category}:")
            for q in questions[:2]:
                print(f"    - {q}")
    
    # 通用问题
    if 'general_questions' in templates:
        print("\n❓ 通用问题:")
        for category, questions in templates['general_questions'].items():
            print(f"  {category}:")
            for q in questions[:2]:
                print(f"    - {q}")

def interactive_mode(inference, templates):
    """交互式模式"""
    print("\n🎯 交互式模式")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'examples' 查看问题示例")
    print("输入 'help' 查看帮助")
    
    while True:
        try:
            # 获取图片路径
            image_path = input("\n📷 请输入图片路径: ").strip()
            
            if image_path.lower() in ['quit', 'exit']:
                print("👋 再见!")
                break
            
            if image_path.lower() == 'examples':
                show_question_examples(templates)
                continue
            
            if image_path.lower() == 'help':
                print("\n📖 帮助:")
                print("- 输入图片路径，然后输入问题")
                print("- 支持的问题类型：营养、食材、健康指标、通用问题")
                print("- 输入 'examples' 查看问题示例")
                print("- 输入 'quit' 退出")
                continue
            
            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"❌ 图片不存在: {image_path}")
                continue
            
            # 获取问题
            question = input("❓ 请输入问题 (直接回车使用默认分析): ").strip()
            
            if not question:
                question = None
            
            # 分析图片
            print(f"\n🔍 分析中...")
            result = inference.analyze_food_image(image_path, question)
            
            if result['success']:
                print("\n" + "="*60)
                print("📊 分析结果:")
                print("="*60)
                print(result['raw_response'])
                
                if result.get('question'):
                    print(f"\n❓ 问题: {result['question']}")
                
                print(f"\n📁 图片: {result['image_path']}")
                print(f"🤖 模型: {result['model_key']}")
            else:
                print(f"❌ 分析失败: {result['raw_response']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def batch_mode(inference, image_dir, question):
    """批量模式"""
    print(f"\n📁 批量分析模式")
    print(f"图片目录: {image_dir}")
    print(f"问题: {question or '默认分析'}")
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("❌ 未找到图片文件")
        return
    
    print(f"📊 找到 {len(image_paths)} 张图片")
    
    # 批量分析
    results = inference.batch_analyze(image_paths, question)
    
    # 保存结果
    output_path = "interactive_results.json"
    inference.save_results(results, output_path)
    
    # 打印统计信息
    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ 分析完成: {successful}/{len(results)} 成功")
    print(f"📄 结果已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Interactive Food VLM Inference")
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g., llava_7b, qwen_vl_7b)")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--image_path", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--question", type=str, help="Question to ask about the image(s)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--mode", choices=["interactive", "single", "batch"], default="interactive", help="运行模式")
    
    args = parser.parse_args()
    
    print("🍽️  Food VLM Interactive Inference")
    print("="*60)
    
    # 加载问答模板
    templates = load_qa_templates()
    
    # 创建推理器
    print(f"🤖 加载模型: {args.model}")
    inference = FoodVLMInference(
        model_key=args.model,
        lora_path=args.lora_path,
        device=args.device
    )
    
    if args.mode == "interactive":
        # 交互式模式
        show_question_examples(templates)
        interactive_mode(inference, templates)
    
    elif args.mode == "single":
        # 单张图片模式
        if not args.image_path:
            print("❌ 单张图片模式需要 --image_path 参数")
            return
        
        print(f"📷 分析图片: {args.image_path}")
        result = inference.analyze_food_image(args.image_path, args.question)
        
        if result['success']:
            print("\n" + "="*60)
            print("📊 分析结果:")
            print("="*60)
            print(result['raw_response'])
            
            if result.get('question'):
                print(f"\n❓ 问题: {result['question']}")
        else:
            print(f"❌ 分析失败: {result['raw_response']}")
    
    elif args.mode == "batch":
        # 批量模式
        if not args.image_dir:
            print("❌ 批量模式需要 --image_dir 参数")
            return
        
        batch_mode(inference, args.image_dir, args.question)

if __name__ == "__main__":
    main()
