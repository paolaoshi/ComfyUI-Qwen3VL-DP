#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型下载功能
Test model download functionality
"""

import sys
from pathlib import Path

# 添加ComfyUI路径到sys.path
comfyui_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(comfyui_path))

import folder_paths
from qwen3vl_node import ModelDownloader, load_model_configs

def test_model_download():
    """测试模型下载功能"""
    print("=" * 60)
    print("🧪 测试 Qwen3VL 模型下载功能")
    print("=" * 60)
    
    # 加载模型配置
    load_model_configs()
    from qwen3vl_node import MODEL_CONFIGS
    
    print(f"\n📋 可用模型列表:")
    for i, model_name in enumerate(MODEL_CONFIGS.keys(), 1):
        if not model_name.startswith('_'):
            model_info = MODEL_CONFIGS[model_name]
            print(f"  {i}. {model_name}")
            print(f"     Repo: {model_info.get('repo_id', 'N/A')}")
    
    # 创建下载器
    downloader = ModelDownloader(MODEL_CONFIGS)
    
    print(f"\n📁 模型存储目录: {downloader.models_dir}")
    print(f"   目录是否存在: {'✅ 是' if downloader.models_dir.exists() else '❌ 否'}")
    
    # 检查已下载的模型
    print(f"\n🔍 检查已下载的模型:")
    if downloader.models_dir.exists():
        downloaded_models = list(downloader.models_dir.iterdir())
        if downloaded_models:
            for model_dir in downloaded_models:
                if model_dir.is_dir():
                    config_file = model_dir / "config.json"
                    model_file = model_dir / "model.safetensors"
                    model_index = model_dir / "model.safetensors.index.json"
                    
                    status = "✅ 完整" if config_file.exists() and (model_file.exists() or model_index.exists()) else "⚠️ 不完整"
                    print(f"  - {model_dir.name}: {status}")
        else:
            print("  ❌ 没有已下载的模型")
    else:
        print("  ❌ 模型目录不存在")
    
    print("\n" + "=" * 60)
    print("💡 提示:")
    print("  - 首次使用时，模型会自动下载")
    print("  - 模型会保存到 ComfyUI/models/prompt_generator/")
    print("  - 如果模型已存在，不会重复下载")
    print("=" * 60)
    
    # 询问是否测试下载
    print("\n🤔 是否要测试下载一个小模型？(Qwen3-VL-2B-Instruct, ~4GB)")
    print("   输入 'yes' 开始下载，其他任意键跳过")
    
    try:
        choice = input(">>> ").strip().lower()
        if choice == 'yes':
            print("\n📥 开始测试下载...")
            try:
                model_path = downloader.ensure_model_available("Qwen3-VL-2B-Instruct")
                print(f"\n✅ 测试成功！模型路径: {model_path}")
            except Exception as e:
                print(f"\n❌ 下载失败: {e}")
        else:
            print("\n⏭️ 跳过下载测试")
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试中断")
    
    print("\n✨ 测试完成！")

if __name__ == "__main__":
    test_model_download()
