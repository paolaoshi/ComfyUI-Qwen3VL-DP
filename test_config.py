#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试配置文件和新模型"""

import json
from pathlib import Path

# 读取配置文件
config_path = Path(__file__).parent / "config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计模型数量
models = [k for k in data.keys() if not k.startswith('_')]
print(f"✅ 配置文件加载成功！")
print(f"📊 共找到 {len(models)} 个模型")

# 检查新模型
new_model_name = 'Huihui-Qwen3-VL-4B-Instruct-Abliterated'
if new_model_name in data:
    model_config = data[new_model_name]
    print(f"\n✅ 新模型 '{new_model_name}' 配置成功！")
    print(f"   📦 Repo ID: {model_config.get('repo_id')}")
    print(f"   🌐 来源: {model_config.get('source', 'huggingface')}")
    print(f"   💾 显存需求: {model_config.get('vram_requirement')}")
    print(f"   ⚠️  警告: {model_config.get('warning', '无')}")
else:
    print(f"\n❌ 未找到新模型 '{new_model_name}'")

# 列出所有模型
print(f"\n📋 所有可用模型:")
for i, model_name in enumerate(models, 1):
    model_info = data[model_name]
    source = model_info.get('source', 'huggingface')
    print(f"  {i}. {model_name} (来源: {source})")
