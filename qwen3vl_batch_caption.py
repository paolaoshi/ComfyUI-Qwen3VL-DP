"""
Qwen3VL 批量打标节点
用于批量处理文件夹中的图像，生成对应的文本描述文件

功能特点：
- 支持批量处理多种图像格式（jpg, jpeg, png, bmp, webp, gif）
- 可选择不同的预设提示词或自定义提示词
- 支持强制覆盖已存在的描述文件（用于重新生成不同风格的描述）
- 支持文件重命名和添加前缀后缀
- 支持连接Qwen3VL额外选项节点，实现模块化的描述控制
- 提供详细的处理日志和错误信息，便于调试

模块化设计：
- 可选连接"Qwen3VL额外选项"节点来精细化控制描述内容
- 保持主节点的简洁性，复杂选项通过独立节点管理
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import comfy.utils

# 动态导入同目录下的 qwen3vl_node 模块
# 避免相对导入问题
try:
    from .qwen3vl_node import (
        Qwen3VL_Advanced, 
        MODEL_CONFIGS, 
        SYSTEM_PROMPTS,
        Quantization,
        ImageProcessor
    )
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import qwen3vl_node
    Qwen3VL_Advanced = qwen3vl_node.Qwen3VL_Advanced
    MODEL_CONFIGS = qwen3vl_node.MODEL_CONFIGS
    SYSTEM_PROMPTS = qwen3vl_node.SYSTEM_PROMPTS
    Quantization = qwen3vl_node.Quantization
    ImageProcessor = qwen3vl_node.ImageProcessor


class Qwen3VL_Batch_Caption:
    """Qwen3-VL 批量打标节点 - 批量处理文件夹中的图像"""
    
    def __init__(self):
        # 复用 Qwen3VL_Advanced 的核心功能
        self.advanced_node = Qwen3VL_Advanced()
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义批量打标节点输入类型"""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["详细描述这张图片"])
        
        return {
            "required": {
                "🤖 模型选择": (model_names, {"default": default_model}),
                "⚙️ 量化级别": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "📁 输入文件夹": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入包含图像的文件夹路径"
                }),
                "📂 输出文件夹": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "留空则保存到输入文件夹"
                }),
                "💭 预设提示词": (preset_prompts, {"default": preset_prompts[2]}),
                "✏️ 自定义提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "可选择预设提示词或输入自定义提示词"
                }),
                "🔢 最大令牌数": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "🌡️ 采样温度": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 核采样参数": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔄 保持模型加载": ("BOOLEAN", {"default": True}),
                "🎲 随机种子": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "📝 前缀文本": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "添加到描述开头的文本"
                }),
                "📌 后缀文本": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "添加到描述结尾的文本"
                }),
                "🔄 重命名文件": ("BOOLEAN", {"default": False}),
                "🏷️ 文件名前缀": ("STRING", {
                    "default": "image_",
                    "multiline": False,
                    "placeholder": "重命名时使用的前缀"
                }),
                "🔢 起始编号": ("INT", {"default": 1, "min": 0, "max": 9999999, "step": 1}),
                "🔄 强制覆盖": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后会覆盖已存在的txt文件，用于重新生成不同风格的描述"
                }),
            },
            "optional": {
                "🎯 Qwen3VL额外选项": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "可选的Qwen3VL额外选项，连接Qwen3VL额外选项节点"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("处理结果",)
    FUNCTION = "batch_process"
    CATEGORY = "🍭大炮-Qwen3VL"
    
    def process_single_image(self, image_path: str, prompt_text: str, **kwargs) -> str:
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            prompt_text: 提示词文本
            **kwargs: 其他参数
            
        Returns:
            生成的描述文本
        """
        try:
            print(f"   🔍 开始处理图像: {os.path.basename(image_path)}")
            # 加载图像
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 转换为tensor格式 (ComfyUI格式: H,W,C, 范围0-1)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # 添加batch维度
                
                # 调用高级节点的处理函数
                result = self.advanced_node.process(
                    **{
                        "🤖 模型选择": kwargs.get("模型名称"),
                        "⚙️ 量化级别": kwargs.get("量化级别"),
                        "💭 预设提示词": kwargs.get("预设提示词"),
                        "✏️ 自定义提示词": prompt_text,
                        "🔢 最大令牌数": kwargs.get("最大令牌数"),
                        "🌡️ 采样温度": kwargs.get("采样温度"),
                        "🎯 核采样参数": kwargs.get("核采样参数"),
                        "🔍 束搜索数量": 1,
                        "🚫 重复惩罚": 1.2,
                        "🎬 视频帧数": 16,
                        "💻 设备选择": "auto",
                        "🔄 保持模型加载": True,  # 批量处理时始终保持加载
                        "🎲 随机种子": kwargs.get("随机种子"),
                        "🖼️ 图像1": img_tensor,
                    }
                )
                
                return result[0] if result else ""
                
        except Exception as e:
            print(f"❌ 处理图像失败 {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈信息
            return ""
    
    @torch.no_grad()
    def batch_process(self, **kwargs):
        """批量处理文件夹中的图像"""
        # 提取参数
        模型名称 = kwargs.get("🤖 模型选择")
        量化级别 = kwargs.get("⚙️ 量化级别")
        输入文件夹 = kwargs.get("📁 输入文件夹", "").strip()
        输出文件夹 = kwargs.get("📂 输出文件夹", "").strip()
        预设提示词 = kwargs.get("💭 预设提示词")
        自定义提示词 = kwargs.get("✏️ 自定义提示词", "").strip()
        最大令牌数 = kwargs.get("🔢 最大令牌数")
        采样温度 = kwargs.get("🌡️ 采样温度")
        核采样参数 = kwargs.get("🎯 核采样参数")
        保持模型加载 = kwargs.get("🔄 保持模型加载")
        随机种子 = kwargs.get("🎲 随机种子")
        前缀文本 = kwargs.get("📝 前缀文本", "").strip()
        后缀文本 = kwargs.get("📌 后缀文本", "").strip()
        重命名文件 = kwargs.get("🔄 重命名文件", False)
        文件名前缀 = kwargs.get("🏷️ 文件名前缀", "image_")
        起始编号 = kwargs.get("🔢 起始编号", 1)
        强制覆盖 = kwargs.get("🔄 强制覆盖", False)
        
        # Qwen3VL 额外选项（可选）
        qwen3vl_extra_options = kwargs.get("🎯 Qwen3VL额外选项", None)
        
        # 验证输入文件夹
        if not 输入文件夹 or not os.path.exists(输入文件夹):
            return ("❌ 错误: 输入文件夹路径无效或不存在",)
        
        # 设置输出文件夹
        if not 输出文件夹:
            输出文件夹 = 输入文件夹
        else:
            os.makedirs(输出文件夹, exist_ok=True)
        
        # 确定使用的提示词
        base_prompt = SYSTEM_PROMPTS.get(预设提示词, 预设提示词)
        if 自定义提示词:
            base_prompt = 自定义提示词
        
        # 应用Qwen3VL额外选项生成增强提示词（如果有的话）
        if qwen3vl_extra_options:
            # 导入Qwen3VL额外选项节点的静态方法
            try:
                from .qwen3vl_extra_options import Qwen3VL_ExtraOptions
                prompt_text = Qwen3VL_ExtraOptions.build_enhanced_prompt(base_prompt, qwen3vl_extra_options)
            except ImportError:
                print("⚠️ 警告: 无法导入Qwen3VL额外选项模块，使用基础提示词")
                prompt_text = base_prompt
        else:
            prompt_text = base_prompt
        
        # 支持的图像格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
        
        # 收集所有图像文件
        image_files = []
        for filename in os.listdir(输入文件夹):
            if filename.lower().endswith(image_extensions):
                image_files.append(filename)
        
        if not image_files:
            return (f"⚠️ 警告: 在文件夹 {输入文件夹} 中未找到图像文件",)
        
        # 排序文件列表
        image_files.sort()
        
        print(f"\n{'='*60}")
        print(f"🚀 开始批量打标")
        print(f"📁 输入文件夹: {输入文件夹}")
        print(f"📂 输出文件夹: {输出文件夹}")
        print(f"🖼️ 图像数量: {len(image_files)}")
        print(f"💭 基础提示词: {base_prompt}")
        
        # 显示Qwen3VL额外选项状态
        if qwen3vl_extra_options:
            enabled_options = [key for key, value in qwen3vl_extra_options.items() if value]
            if enabled_options:
                print(f"🎯 Qwen3VL额外选项: 已启用 ({len(enabled_options)}个)")
                print(f"   启用的选项: {', '.join(enabled_options)}")
            else:
                print(f"🎯 Qwen3VL额外选项: 已连接但未启用任何选项")
        else:
            print(f"🎯 Qwen3VL额外选项: 未连接")
        
        print(f"🔄 强制覆盖: {'是' if 强制覆盖 else '否'}")
        print(f"📋 找到的图像文件:")
        for i, file in enumerate(image_files, 1):
            print(f"   {i}. {file}")
        print(f"{'='*60}\n")
        
        # 统计信息
        成功数量 = 0
        失败数量 = 0
        跳过数量 = 0
        开始时间 = time.time()
        
        # 记录处理结果的详细信息
        成功文件 = []
        失败文件 = []
        跳过文件 = []
        
        # 创建进度条
        pbar = comfy.utils.ProgressBar(len(image_files))
        
        # 处理每张图像
        当前编号 = 起始编号
        for idx, filename in enumerate(image_files):
            try:
                image_path = os.path.join(输入文件夹, filename)
                base_name = os.path.splitext(filename)[0]
                
                # 确定输出文件名
                if 重命名文件:
                    new_base_name = f"{文件名前缀}{当前编号:04d}"
                    当前编号 += 1
                else:
                    new_base_name = base_name
                    # 如果文件名重复，添加扩展名后缀以区分
                    original_ext = os.path.splitext(filename)[1].lower()
                    if original_ext in ['.jpeg', '.jpg']:
                        ext_suffix = '_jpg'
                    elif original_ext == '.png':
                        ext_suffix = '_png'
                    elif original_ext == '.bmp':
                        ext_suffix = '_bmp'
                    elif original_ext == '.webp':
                        ext_suffix = '_webp'
                    elif original_ext == '.gif':
                        ext_suffix = '_gif'
                    else:
                        ext_suffix = original_ext.replace('.', '_')
                    
                    # 检查是否需要添加后缀来避免重复
                    base_text_path = os.path.join(输出文件夹, f"{new_base_name}.txt")
                    if os.path.exists(base_text_path) and not 强制覆盖:
                        new_base_name = f"{base_name}{ext_suffix}"
                
                text_path = os.path.join(输出文件夹, f"{new_base_name}.txt")
                
                # 检查是否已存在描述文件（只有在不强制覆盖时才跳过）
                if os.path.exists(text_path) and not 强制覆盖:
                    print(f"⏭️ 跳过已存在: {filename} (如需覆盖请启用'强制覆盖'选项)")
                    跳过数量 += 1
                    跳过文件.append(filename)
                    pbar.update_absolute(idx + 1, len(image_files))
                    continue
                elif os.path.exists(text_path) and 强制覆盖:
                    print(f"🔄 强制覆盖: {filename}")
                
                print(f"🖼️ 处理中 [{idx+1}/{len(image_files)}]: {filename}")
                print(f"   📁 图像路径: {image_path}")
                print(f"   📝 输出路径: {text_path}")
                
                # 处理图像
                caption = self.process_single_image(
                    image_path,
                    prompt_text,
                    模型名称=模型名称,
                    量化级别=量化级别,
                    预设提示词=预设提示词,
                    最大令牌数=最大令牌数,
                    采样温度=采样温度,
                    核采样参数=核采样参数,
                    随机种子=随机种子
                )
                
                if caption:
                    # 添加前缀和后缀
                    if 前缀文本:
                        caption = f"{前缀文本} {caption}"
                    if 后缀文本:
                        caption = f"{caption} {后缀文本}"
                    
                    # 保存描述文件
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    
                    print(f"✅ 成功: {new_base_name}.txt")
                    print(f"   描述: {caption[:100]}{'...' if len(caption) > 100 else ''}\n")
                    成功数量 += 1
                    成功文件.append(filename)
                else:
                    print(f"❌ 失败: 生成描述为空\n")
                    失败数量 += 1
                    失败文件.append(filename)
                
                # 更新进度条
                pbar.update_absolute(idx + 1, len(image_files))
                
            except Exception as e:
                print(f"❌ 处理失败 {filename}: {str(e)}\n")
                失败数量 += 1
                失败文件.append(filename)
                pbar.update_absolute(idx + 1, len(image_files))
        
        # 清理模型（如果不保持加载）
        if not 保持模型加载:
            self.advanced_node.clear_model_resources()
        
        # 计算总耗时
        总耗时 = time.time() - 开始时间
        平均耗时 = 总耗时 / len(image_files) if image_files else 0
        
        # 生成详细的结果报告
        详细报告 = []
        
        if 成功文件:
            详细报告.append("✅ 成功处理的文件:")
            for file in 成功文件:
                详细报告.append(f"   • {file}")
        
        if 失败文件:
            详细报告.append("❌ 处理失败的文件:")
            for file in 失败文件:
                详细报告.append(f"   • {file}")
        
        if 跳过文件:
            详细报告.append("⏭️ 跳过的文件:")
            for file in 跳过文件:
                详细报告.append(f"   • {file}")
        
        详细信息 = "\n".join(详细报告) if 详细报告 else ""
        
        # 生成结果报告
        结果报告 = f"""
{'='*60}
📊 批量打标完成
{'='*60}
✅ 成功: {成功数量} 张
❌ 失败: {失败数量} 张
⏭️ 跳过: {跳过数量} 张
📁 总计: {len(image_files)} 张
⏱️ 总耗时: {总耗时:.2f} 秒
⚡ 平均耗时: {平均耗时:.2f} 秒/张
📂 输出位置: {输出文件夹}
{'='*60}

{详细信息}

{'='*60}
"""
        
        print(结果报告)
        
        return (结果报告,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Batch_Caption": Qwen3VL_Batch_Caption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Batch_Caption": "🍭大炮-Qwen3VL批量打标@炮老师的小课堂",
}
