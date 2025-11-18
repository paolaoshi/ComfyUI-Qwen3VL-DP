# ModelScope 模型使用说明

## 📋 概述

本项目现已支持从 ModelScope 下载社区模型。ModelScope 是一个中文友好的模型托管平台，访问速度通常比 HuggingFace 更快。

## 🆕 新增模型

### Huihui-Qwen3-VL-4B-Instruct-Abliterated

- **模型来源**: ModelScope
- **仓库地址**: https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated
- **基础模型**: Qwen3-VL-4B-Instruct
- **特点**: 已移除安全过滤（abliterated）

#### 显存需求
- **完整精度 (FP16)**: 6GB
- **8-bit 量化**: 3.5GB
- **4-bit 量化**: 2GB

#### ⚠️ 重要警告
此模型已移除安全过滤机制，可能生成敏感或不当内容。请仅在以下场景使用：
- 研究和学术用途
- 受控的测试环境
- 了解风险并能承担责任的场景

**不建议在生产环境或面向公众的应用中使用此模型。**

## 📦 安装依赖

### 自动安装（推荐）

首次使用 ModelScope 模型时，系统会自动提示安装依赖：

```bash
pip install modelscope
```

### 手动安装

如果需要提前安装，可以运行：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```

`requirements.txt` 已包含 `modelscope` 依赖。

## 🚀 使用方法

### 1. 在节点中选择模型

在任何 Qwen3VL 节点（主节点、批量打标、对比打标等）的模型下拉列表中，选择：

```
Huihui-Qwen3-VL-4B-Instruct-Abliterated
```

### 2. 自动下载

首次使用时，模型会自动从 ModelScope 下载到：

```
ComfyUI/models/prompt_generator/Huihui-Qwen3-VL-4B-Instruct-abliterated/
```

### 3. 下载过程

```
📥 正在从 MODELSCOPE 下载模型 'Huihui-Qwen3-VL-4B-Instruct-Abliterated' 到 ...
📁 目标路径: ComfyUI/models/prompt_generator/Huihui-Qwen3-VL-4B-Instruct-abliterated/
⏳ 提示：首次下载可能需要较长时间，请耐心等待...
```

### 4. 后续使用

模型下载完成后，下次使用会直接加载，不会重复下载：

```
✅ 模型 'Huihui-Qwen3-VL-4B-Instruct-Abliterated' 已存在于 ...
📁 模型路径: ComfyUI/models/prompt_generator/Huihui-Qwen3-VL-4B-Instruct-abliterated/
```

## 🔧 技术细节

### 多源支持

项目现在支持两种模型来源：

1. **HuggingFace** (默认)
   - 官方 Qwen 模型
   - 大部分社区模型

2. **ModelScope** (新增)
   - 中文社区模型
   - 国内访问速度更快
   - 需要安装 `modelscope` 库

### 配置文件

模型来源在 `config.json` 中配置：

```json
{
  "Huihui-Qwen3-VL-4B-Instruct-Abliterated": {
    "repo_id": "fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "source": "modelscope",  // 指定来源为 ModelScope
    "default": false,
    "quantized": false,
    "abliterated": true,
    "vram_requirement": {
      "full": 6.0,
      "8bit": 3.5,
      "4bit": 2.0
    },
    "warning": "此模型已移除安全过滤，可能生成敏感内容。仅用于研究和测试环境。"
  }
}
```

### 下载逻辑

代码会根据 `source` 字段自动选择下载方式：

```python
# 检查模型来源
source = model_info.get('source', 'huggingface')

# 根据来源选择下载函数
if source == 'modelscope':
    # 使用 ModelScope 下载
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(model_id=repo_id, ...)
else:
    # 使用 HuggingFace 下载
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, ...)
```

## ❓ 常见问题

### Q1: ModelScope 库未安装怎么办？

**A**: 运行以下命令安装：

```bash
pip install modelscope
```

或者重新安装项目依赖：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```

### Q2: 下载失败怎么办？

**A**: 如果自动下载失败，可以：

1. 检查网络连接
2. 手动从 ModelScope 下载模型文件
3. 将文件放到：`ComfyUI/models/prompt_generator/Huihui-Qwen3-VL-4B-Instruct-abliterated/`

手动下载地址：
https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated/files

### Q3: 如何添加更多 ModelScope 模型？

**A**: 编辑 `config.json`，添加新模型配置：

```json
{
  "你的模型名称": {
    "repo_id": "modelscope上的仓库ID",
    "source": "modelscope",
    "default": false,
    "quantized": false,
    "vram_requirement": {
      "full": 6.0,
      "8bit": 3.5,
      "4bit": 2.0
    }
  }
}
```

### Q4: ModelScope 和 HuggingFace 有什么区别？

**A**: 
- **ModelScope**: 中文平台，国内访问速度快，部分社区模型
- **HuggingFace**: 国际平台，模型最全，但国内访问可能较慢

项目会根据模型配置自动选择合适的下载源。

## 📚 相关链接

- **ModelScope 官网**: https://modelscope.cn
- **模型仓库**: https://modelscope.cn/models/fireicewolf/Huihui-Qwen3-VL-4B-Instruct-abliterated
- **项目文档**: README.md
- **更新日志**: CHANGELOG_修改说明.md

## 🤝 贡献

如果你有好的 ModelScope 模型推荐，欢迎提交 Issue 或 Pull Request！
