# 🗂️ 项目文件结构

## 📁 核心文件

### 🔧 配置文件
- `__init__.py` - 节点自动加载和注册
- `config.json` - 模型配置和预设提示词
- `requirements.txt` - Python依赖包列表

### 🎮 节点文件
- `qwen3vl_node.py` - 核心Qwen3VL节点（简单版和高级版）
- `qwen3vl_batch_caption.py` - 批量打标节点
- `qwen3vl_compare_caption.py` - 对比打标节点
- `qwen3vl_extra_options.py` - 额外选项配置节点

### 📚 文档文件
- `README.md` - 中文项目说明文档
- `README_EN.md` - 英文项目说明文档
- `PROJECT_STRUCTURE.md` - 项目文件结构说明

## 🎯 节点功能映射

### 主要节点
1. **🍭大炮-Qwen3VL (简单)** → `qwen3vl_node.py:Qwen3VL_Simple`
2. **🍭大炮-Qwen3VL (高级)** → `qwen3vl_node.py:Qwen3VL_Advanced`

### 批量处理节点
3. **🍭大炮-Qwen3VL批量打标** → `qwen3vl_batch_caption.py:Qwen3VL_Batch_Caption`
4. **🍭大炮-Qwen3VL对比打标** → `qwen3vl_compare_caption.py:Qwen3VL_Compare_Caption`

### 配置节点
5. **🍭大炮-Qwen3VL额外选项** → `qwen3vl_extra_options.py:Qwen3VL_ExtraOptions`

## 📊 文件大小统计

| 文件名 | 大小 | 功能描述 |
|--------|------|----------|
| `qwen3vl_node.py` | ~33KB | 核心模型处理逻辑 |
| `qwen3vl_compare_caption.py` | ~19KB | 对比打标功能 |
| `qwen3vl_batch_caption.py` | ~18KB | 批量打标功能 |
| `README.md` | ~12KB | 中文文档 |
| `qwen3vl_extra_options.py` | ~10KB | 额外选项配置 |
| `README_EN.md` | ~9KB | 英文文档 |
| `config.json` | ~6KB | 配置文件 |
| `__init__.py` | ~2KB | 初始化文件 |
| `requirements.txt` | ~0.4KB | 依赖列表 |

## 🔗 依赖关系

```
qwen3vl_node.py (核心)
    ↑
    ├── qwen3vl_batch_caption.py (依赖核心)
    ├── qwen3vl_compare_caption.py (依赖核心)
    └── qwen3vl_extra_options.py (独立，被批量打标调用)
```

## 🎨 代码架构特点

### 模块化设计
- **核心分离**: 主要模型逻辑在 `qwen3vl_node.py`
- **功能扩展**: 批量和对比功能作为独立模块
- **配置解耦**: 额外选项作为可选模块

### 代码复用
- 批量打标和对比打标都复用核心节点的 `Qwen3VL_Advanced` 类
- 统一的错误处理和进度显示机制
- 共享的模型配置和提示词系统

### 中文友好
- 全中文参数名（使用emoji图标）
- 详细的中文注释和文档字符串
- 面向中文用户的错误提示和日志

## 🚀 部署清单

发布前确认以下文件：
- [x] `qwen3vl_node.py` - 核心功能
- [x] `qwen3vl_batch_caption.py` - 批量打标
- [x] `qwen3vl_compare_caption.py` - 对比打标  
- [x] `qwen3vl_extra_options.py` - 额外选项
- [x] `__init__.py` - 节点注册
- [x] `config.json` - 配置文件
- [x] `requirements.txt` - 依赖列表
- [x] `README.md` - 中文文档
- [x] `README_EN.md` - 英文文档
- [x] `PROJECT_STRUCTURE.md` - 项目结构说明

## 📝 版本信息
- **当前版本**: v2.0.0
- **发布日期**: 2025-11-14
- **主要特性**: 批量打标、对比打标、模块化设计
