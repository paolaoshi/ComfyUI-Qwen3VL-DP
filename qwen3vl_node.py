import torch
import time
import json
import platform
import psutil
import numpy as np
from PIL import Image
from enum import Enum
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download as hf_snapshot_download
import folder_paths
import gc

# 尝试导入 ModelScope，如果不存在则使用 HuggingFace
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    ms_snapshot_download = None
    MODELSCOPE_AVAILABLE = False
    print("[Qwen3VL] ⚠️ ModelScope 未安装，ModelScope 模型将无法下载。请运行: pip install modelscope")

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}

def load_model_configs():
    """加载模型配置文件"""
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            MODEL_CONFIGS = json.load(f)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 {CONFIG_PATH}")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}
    except json.JSONDecodeError:
        print(f"错误: 配置文件解析失败")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}

    # 加载用户自定义模型配置
    custom_path = NODE_DIR / "custom_models.json"
    if custom_path.exists():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_data = json.load(f) or {}
            
            user_models = custom_data.get("hf_models", {}) or custom_data.get("models", {})
            
            if user_models:
                MODEL_CONFIGS.update(user_models)
                print(f"[Qwen3VL] ✅ 已加载 {len(user_models)} 个自定义模型")
            else:
                print("[Qwen3VL] ⚠️ 找到 custom_models.json 但没有有效的模型条目")
        except Exception as e:
            print(f"[Qwen3VL] ⚠️ 加载 custom_models.json 失败 → {e}")
    else:
        print("[Qwen3VL] ℹ️ 未找到 custom_models.json，跳过自定义模型")

if not MODEL_CONFIGS:
    load_model_configs()

class Quantization(str, Enum):
    """量化选项枚举"""
    Q4_BIT = "4-bit (节省显存)"
    Q8_BIT = "8-bit (平衡)"
    NONE = "None (FP16)"
    
    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

def get_model_info(model_name: str) -> dict:
    """获取模型信息"""
    return MODEL_CONFIGS.get(model_name, {})

def get_device_info() -> dict:
    """获取设备信息"""
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {
            "available": True,
            "total_memory": total_mem,
            "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3)
        }
    else:
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {
        "total": sys_mem.total / 1024**3,
        "available": sys_mem.available / 1024**3
    }

    device_info = {
        "gpu": gpu_info,
        "system_memory": sys_mem_info,
        "device_type": "cpu",
        "recommended_device": "cpu",
        "memory_sufficient": True,
        "warning_message": ""
    }

    if platform.system() == "Darwin" and platform.processor() == "arm":
        device_info.update({
            "device_type": "apple_silicon",
            "recommended_device": "mps"
        })
        if sys_mem_info["total"] < 16:
            device_info.update({
                "memory_sufficient": False,
                "warning_message": "Apple Silicon 内存小于 16GB，性能可能受影响"
            })
    elif gpu_info["available"]:
        device_info.update({
            "device_type": "nvidia_gpu",
            "recommended_device": "cuda"
        })
        if gpu_info["total_memory"] < 8:
            device_info.update({
                "memory_sufficient": False,
                "warning_message": "GPU 显存小于 8GB，性能可能下降"
            })
    
    return device_info

def check_memory_requirements(model_name: str, quantization: str, device_info: dict) -> str:
    """检查内存需求并自动调整量化级别"""
    model_info = get_model_info(model_name)
    vram_req = model_info.get("vram_requirement", {})
    
    quant_map = {
        Quantization.Q4_BIT: vram_req.get("4bit", 0),
        Quantization.Q8_BIT: vram_req.get("8bit", 0),
        Quantization.NONE: vram_req.get("full", 0)
    }
    
    base_memory = quant_map.get(quantization, 0)
    device = device_info["recommended_device"]
    use_cpu_mps = device in ["cpu", "mps"]
    
    required_mem = base_memory * (1.5 if use_cpu_mps else 1.0)
    available_mem = device_info["system_memory"]["available"] if use_cpu_mps else device_info["gpu"]["free_memory"]
    mem_type = "系统内存" if use_cpu_mps else "GPU显存"

    if required_mem * 1.2 > available_mem:
        print(f"警告: {mem_type} 不足 ({available_mem:.2f}GB 可用)。降低量化级别...")
        if quantization == Quantization.NONE:
            return Quantization.Q8_BIT
        if quantization == Quantization.Q8_BIT:
            return Quantization.Q4_BIT
        raise RuntimeError(f"{mem_type} 不足，即使使用 4-bit 量化也无法运行")
    
    return quantization

def check_flash_attention() -> bool:
    """检查是否支持 Flash Attention 2"""
    try:
        import flash_attn
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    except ImportError:
        return False
    return False

class ImageProcessor:
    """图像处理器"""
    def to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """将 ComfyUI 图像张量转换为 PIL Image"""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

class ModelDownloader:
    """模型下载器
    
    模型存储路径：ComfyUI/models/prompt_generator/
    """
    def __init__(self, configs):
        self.configs = configs
        # 修改模型存储路径为 prompt_generator 文件夹
        self.models_dir = Path(folder_paths.models_dir) / "prompt_generator"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def ensure_model_available(self, model_name):
        """确保模型可用，如果不存在则下载
        
        模型会直接下载到 ComfyUI/models/prompt_generator/ 目录
        如果模型已存在，则直接使用，不会重复下载
        支持 HuggingFace 和 ModelScope 两种来源
        """
        model_info = self.configs.get(model_name)
        if not model_info:
            raise ValueError(f"模型 '{model_name}' 未在配置中找到")

        repo_id = model_info['repo_id']
        source = model_info.get('source', 'huggingface')  # 默认使用 HuggingFace
        model_folder_name = repo_id.split('/')[-1]
        model_path = self.models_dir / model_folder_name
        
        # 检查模型是否已完整下载（检查关键文件是否存在）
        config_file = model_path / "config.json"
        model_file = model_path / "model.safetensors"
        # 有些模型使用分片存储
        model_index = model_path / "model.safetensors.index.json"
        
        if model_path.exists() and config_file.exists():
            # 检查模型文件是否存在（完整模型或分片模型）
            if model_file.exists() or model_index.exists():
                print(f"✅ 模型 '{model_name}' 已存在于 {model_path}")
                print(f"📁 模型路径: {model_path}")
                return str(model_path)
            else:
                print(f"⚠️ 模型目录存在但文件不完整，将重新下载...")
        
        # 检查 ModelScope 模型是否需要安装依赖
        if source == 'modelscope' and not MODELSCOPE_AVAILABLE:
            raise RuntimeError(
                f"模型 '{model_name}' 来自 ModelScope，但 ModelScope 库未安装。\n"
                f"请运行以下命令安装：\n"
                f"pip install modelscope\n"
                f"或者使用 HuggingFace 镜像站手动下载模型到: {model_path}"
            )
        
        print(f"📥 正在从 {source.upper()} 下载模型 '{model_name}' 到 {model_path}...")
        print(f"📁 目标路径: {model_path}")
        print("⏳ 提示：首次下载可能需要较长时间，请耐心等待...")
        
        # 创建模型目录
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 根据来源选择下载函数
        if source == 'modelscope':
            snapshot_download_func = ms_snapshot_download
            download_kwargs = {
                "model_id": repo_id,
                "cache_dir": str(model_path.parent),
                "local_dir": str(model_path),
            }
            source_url = f"https://modelscope.cn/models/{repo_id}"
        else:
            snapshot_download_func = hf_snapshot_download
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(model_path),
                "local_dir_use_symlinks": False,
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"],
                "resume_download": True,
                "max_workers": 4
            }
            source_url = f"https://huggingface.co/{repo_id}"
        
        # 添加重试机制，解决网络连接问题
        max_retries = 3
        for attempt in range(max_retries):
            try:
                downloaded_path = snapshot_download_func(**download_kwargs)
                print(f"✅ 模型 '{model_name}' 下载完成！")
                print(f"📁 模型已保存到: {model_path}")
                return str(model_path)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ 下载失败（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                    print(f"⏳ 等待 5 秒后重试...")
                    time.sleep(5)
                else:
                    print(f"❌ 下载失败，已重试 {max_retries} 次")
                    error_msg = f"模型下载失败: {str(e)}\n建议：\n"
                    if source == 'modelscope':
                        error_msg += (
                            f"1. 检查网络连接是否正常\n"
                            f"2. 确保已安装 ModelScope: pip install modelscope\n"
                            f"3. 手动从 {source_url} 下载模型到: {model_path}\n"
                        )
                    else:
                        error_msg += (
                            f"1. 检查网络连接是否正常\n"
                            f"2. 设置 HF_ENDPOINT 环境变量使用镜像源（如：https://hf-mirror.com）\n"
                            f"3. 手动从 {source_url} 下载模型到: {model_path}\n"
                        )
                    raise RuntimeError(error_msg)

class Qwen3VL_Advanced:
    """Qwen3-VL 高级节点 - 支持图像和视频理解"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"Qwen3VL 节点已初始化。设备: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"警告: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        """清理模型资源"""
        if self.model is not None:
            print("释放模型资源...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto"):
        """加载模型
        
        Args:
            model_name: 模型名称
            quantization_str: 量化级别字符串
            device: 设备类型 (auto/cuda/cpu/mps)
        
        Raises:
            ValueError: 当 GPU 不支持 FP8 模型或使用 abliterated 模型时
        """
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        
        # 如果模型已加载且配置相同，则跳过
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization_str and 
            self.current_device == effective_device):
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        
        # 检查 abliterated 模型的警告
        if model_info.get("abliterated"):
            warning_msg = model_info.get("warning", "此模型已移除安全过滤")
            print(f"\n⚠️  警告: {warning_msg}\n")
        
        # 检查 FP8 量化模型的 GPU 计算能力要求
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 模型需要计算能力 8.9 或更高的 GPU (例如 RTX 4090)。"
                        f"您的 GPU 计算能力为 {cc}。请选择非 FP8 模型。"
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)
        
        quant_config, load_dtype = None, torch.float16
        
        # 仅对非预量化模型应用量化配置
        if not get_model_info(model_name).get("quantized", False):
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_dtype = None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_dtype = None

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}

        # 构建模型加载参数
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,
            "attn_implementation": "flash_attention_2" if check_flash_attention() else "sdpa",
            "use_safetensors": True,
            "trust_remote_code": True  # abliterated 模型需要
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"正在加载模型 '{model_name}'...")
        # 加载模型、处理器和分词器
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_model_name = model_name
        self.current_quantization = quantization_str
        self.current_device = effective_device
        print("模型加载成功")

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入类型"""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["详细描述这张图片"])

        return {
            "required": {
                "🤖 模型选择": (model_names, {"default": default_model}),
                "⚙️ 量化级别": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "💭 预设提示词": (preset_prompts, {"default": preset_prompts[2]}),
                "✏️ 自定义提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "可选择预设提示词或输入自定义提示词"
                }),
                "🔢 最大令牌数": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "🌡️ 采样温度": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 核采样参数": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔍 束搜索数量": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "🚫 重复惩罚": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎬 视频帧数": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "💻 设备选择": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "🔄 保持模型加载": ("BOOLEAN", {"default": False}),
                "🎲 随机种子": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
                "🎮 种子控制": (["随机", "固定"], {"default": "随机"}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
                "🎥 视频": ("IMAGE",),
                "🎯 Qwen3VL额外选项": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "可选的Qwen3VL额外选项，连接Qwen3VL额外选项节点"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本输出",)
    FUNCTION = "process"
    CATEGORY = "🍭大炮-Qwen3VL"

    @torch.no_grad()
    def process(self, **kwargs):
        """处理图像或视频输入并生成文本 - 使用kwargs处理带emoji的参数名"""
        # 提取参数（兼容带emoji的参数名）
        模型名称 = kwargs.get("🤖 模型选择")
        量化级别 = kwargs.get("⚙️ 量化级别")
        预设提示词 = kwargs.get("💭 预设提示词")
        最大令牌数 = kwargs.get("🔢 最大令牌数")
        采样温度 = kwargs.get("🌡️ 采样温度")
        核采样参数 = kwargs.get("🎯 核采样参数")
        重复惩罚 = kwargs.get("🚫 重复惩罚")
        束搜索数量 = kwargs.get("🔍 束搜索数量")
        视频帧数 = kwargs.get("🎬 视频帧数")
        设备选择 = kwargs.get("💻 设备选择")
        随机种子 = kwargs.get("🎲 随机种子")
        自定义提示词 = kwargs.get("✏️ 自定义提示词", "")
        图像1 = kwargs.get("🖼️ 图像1")
        图像2 = kwargs.get("🖼️ 图像2")
        图像3 = kwargs.get("🖼️ 图像3")
        图像4 = kwargs.get("🖼️ 图像4")
        视频 = kwargs.get("🎥 视频")
        保持模型加载 = kwargs.get("🔄 保持模型加载", True)
        种子控制 = kwargs.get("🎮 种子控制", "随机")
        extra_options = kwargs.get("🎯 Qwen3VL额外选项", None)
        start_time = time.time()
        
        # 根据种子控制设置随机种子
        if 种子控制 == "固定":
            torch.manual_seed(随机种子)
        else:
            torch.manual_seed(int(time.time()))
        
        try:
            self.load_model(模型名称, 量化级别, 设备选择)
            effective_device = self.current_device
            
            # 确定使用的提示词（图像/视频反推专用）
            prompt_text = SYSTEM_PROMPTS.get(预设提示词, 预设提示词)
            if 自定义提示词 and 自定义提示词.strip():
                prompt_text = 自定义提示词.strip()
            
            # 应用Qwen3VL额外选项生成增强提示词（如果有的话）
            if extra_options:
                try:
                    import qwen3vl_extra_options
                    prompt_text = qwen3vl_extra_options.Qwen3VL_ExtraOptions.build_enhanced_prompt(prompt_text, extra_options)
                    print(f"✅ 已应用Qwen3VL额外选项增强提示词")
                except (ImportError, AttributeError) as e:
                    print(f"⚠️ 警告: 无法导入Qwen3VL额外选项模块 ({e})，使用基础提示词")
            
            # 构建对话消息
            conversation = [{"role": "user", "content": []}]
            
            # 添加多个图像
            for i, image in enumerate([图像1, 图像2, 图像3, 图像4], 1):
                if image is not None:
                    conversation[0]["content"].append({
                        "type": "image",
                        "image": self.image_processor.to_pil(image)
                    })
            
            # 添加视频（作为多帧图像序列）
            if 视频 is not None:
                video_frames = [
                    Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8))
                    for frame in 视频
                ]
                
                # 采样视频帧
                if len(video_frames) > 视频帧数:
                    indices = np.linspace(0, len(video_frames) - 1, 视频帧数, dtype=int)
                    sampled_frames = [video_frames[i] for i in indices]
                else:
                    sampled_frames = video_frames

                # 确保至少有2帧（Qwen3-VL 要求）
                if sampled_frames and len(sampled_frames) == 1:
                    sampled_frames.append(sampled_frames[0])
                    
                if sampled_frames:
                    conversation[0]["content"].append({
                        "type": "video",
                        "video": sampled_frames
                    })

            # 添加文本提示
            conversation[0]["content"].append({
                "type": "text",
                "text": prompt_text
            })

            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 提取图像和视频用于处理器
            pil_images = [
                item['image'] for item in conversation[0]['content']
                if item['type'] == 'image'
            ]
            video_frames_list = [
                frame for item in conversation[0]['content']
                if item['type'] == 'video'
                for frame in item['video']
            ]
            videos_arg = [video_frames_list] if video_frames_list else None
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt,
                images=pil_images if pil_images else None,
                videos=videos_arg,
                return_tensors="pt"
            )
            
            # 将输入移到设备
            model_inputs = {
                k: v.to(effective_device)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            # 设置停止标记
            stop_tokens = [self.tokenizer.eos_token_id]
            if hasattr(self.tokenizer, 'eot_id'):
                stop_tokens.append(self.tokenizer.eot_id)

            # 生成参数
            gen_kwargs = {
                "max_new_tokens": 最大令牌数,
                "repetition_penalty": 重复惩罚,
                "num_beams": 束搜索数量,
                "eos_token_id": stop_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            if 束搜索数量 > 1:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": 采样温度,
                    "top_p": 核采样参数
                })

            # 生成文本
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            input_ids_len = model_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(
                outputs[0, input_ids_len:],
                skip_special_tokens=True
            )
            
            print(f"生成完成，耗时 {time.time() - start_time:.2f} 秒")
            return (text.strip(),)

        except (ValueError, RuntimeError) as e:
            error_message = f"错误: {str(e)}"
            print(error_message)
            return (error_message,)
        finally:
            if not 保持模型加载:
                self.clear_model_resources()


class Qwen3VL_Chat:
    """Qwen3-VL 智能对话节点 - 支持多模态LLM对话"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"Qwen3VL 智能对话节点已初始化。设备: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"警告: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        """清理模型资源"""
        if self.model is not None:
            print("释放模型资源...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto"):
        """加载模型"""
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        
        # 如果模型已加载且配置相同，则跳过
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization_str and 
            self.current_device == effective_device):
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        
        # 检查 abliterated 模型的警告
        if model_info.get("abliterated"):
            warning_msg = model_info.get("warning", "此模型已移除安全过滤")
            print(f"\n⚠️  警告: {warning_msg}\n")
        
        # 检查 FP8 量化模型的 GPU 计算能力要求
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 模型需要计算能力 8.9 或更高的 GPU (例如 RTX 4090)。"
                        f"您的 GPU 计算能力为 {cc}。请选择非 FP8 模型。"
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)
        
        quant_config, load_dtype = None, torch.float16
        
        # 仅对非预量化模型应用量化配置
        if not get_model_info(model_name).get("quantized", False):
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_dtype = None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_dtype = None

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}

        # 构建模型加载参数
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,
            "attn_implementation": "flash_attention_2" if check_flash_attention() else "sdpa",
            "use_safetensors": True,
            "trust_remote_code": True
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"正在加载模型 '{model_name}'...")
        # 加载模型、处理器和分词器
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_model_name = model_name
        self.current_quantization = quantization_str
        self.current_device = effective_device
        print("模型加载成功")

    @classmethod
    def INPUT_TYPES(cls):
        """定义智能对话节点输入类型"""
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]

        return {
            "required": {
                "🤖 模型选择": (model_names, {"default": default_model}),
                "⚙️ 量化级别": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "💬 用户输入": ("STRING", {
                    "default": "你好，请介绍一下你自己。",
                    "multiline": True,
                    "placeholder": "输入你想要对话的内容"
                }),
                "🎭 系统角色定义": ("STRING", {
                    "default": "你是一个专业、友好且乐于助人的AI助手。",
                    "multiline": True,
                    "placeholder": "定义AI的角色和行为方式"
                }),
                "🌡️ 温度": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "🎯 Top-P": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "📏 最大长度": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 16}),
                "🎲 随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "🎮 种子控制": (["随机", "固定"], {"default": "随机"}),
                "🔄 保持模型加载": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "🖼️ 图像1": ("IMAGE",),
                "🖼️ 图像2": ("IMAGE",),
                "🖼️ 图像3": ("IMAGE",),
                "🖼️ 图像4": ("IMAGE",),
                "🎯 Qwen3VL额外选项": ("QWEN3VL_EXTRA_OPTIONS", {
                    "tooltip": "可选的Qwen3VL额外选项，连接Qwen3VL额外选项节点"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI回复",)
    FUNCTION = "chat"
    CATEGORY = "🍭大炮-Qwen3VL"

    @torch.no_grad()
    def chat(self, **kwargs):
        """智能对话处理函数"""
        # 提取参数（兼容带emoji的参数名）
        模型名称 = kwargs.get("🤖 模型选择")
        量化级别 = kwargs.get("⚙️ 量化级别")
        用户输入 = kwargs.get("💬 用户输入")
        系统角色定义 = kwargs.get("🎭 系统角色定义")
        温度 = kwargs.get("🌡️ 温度")
        最大长度 = kwargs.get("📏 最大长度")
        随机种子 = kwargs.get("🎲 随机种子")
        种子控制 = kwargs.get("🎮 种子控制")
        保持模型加载 = kwargs.get("🔄 保持模型加载")
        图像1 = kwargs.get("🖼️ 图像1")
        图像2 = kwargs.get("🖼️ 图像2")
        图像3 = kwargs.get("🖼️ 图像3")
        图像4 = kwargs.get("🖼️ 图像4")
        extra_options = kwargs.get("🎯 Qwen3VL额外选项", None)
        # 处理Top-P参数（兼容新旧版本）
        top_p = kwargs.get("🎯 Top-P", 0.90)
        
        start_time = time.time()
        
        # 根据种子控制设置随机种子
        if 种子控制 == "固定":
            torch.manual_seed(随机种子)
        else:
            torch.manual_seed(int(time.time()))
        
        try:
            self.load_model(模型名称, 量化级别, "auto")
            effective_device = self.current_device
            
            # 处理系统角色定义，应用额外选项
            system_prompt = 系统角色定义.strip() if 系统角色定义 else ""
            
            # 应用Qwen3VL额外选项增强系统提示词（如果有的话）
            if extra_options and system_prompt:
                try:
                    import qwen3vl_extra_options
                    system_prompt = qwen3vl_extra_options.Qwen3VL_ExtraOptions.build_enhanced_prompt(system_prompt, extra_options)
                    print(f"✅ 已应用Qwen3VL额外选项增强系统角色")
                except (ImportError, AttributeError) as e:
                    print(f"⚠️ 警告: 无法导入Qwen3VL额外选项模块 ({e})，使用基础系统角色")
            
            # 构建对话消息，先添加系统角色定义
            conversation = []
            
            # 添加系统角色定义（如果提供）
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            # 添加用户消息
            user_content = []
            
            # 添加多个图像
            for i, image in enumerate([图像1, 图像2, 图像3, 图像4], 1):
                if image is not None:
                    user_content.append({
                        "type": "image",
                        "image": self.image_processor.to_pil(image)
                    })
            
            # 添加用户文本输入
            user_content.append({
                "type": "text",
                "text": 用户输入
            })
            
            conversation.append({
                "role": "user",
                "content": user_content
            })

            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 提取图像用于处理器
            pil_images = []
            for msg in conversation:
                if msg['role'] == 'user':
                    pil_images.extend([
                        item['image'] for item in msg['content']
                        if item['type'] == 'image'
                    ])
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt,
                images=pil_images if pil_images else None,
                return_tensors="pt"
            )
            
            # 将输入移到设备
            model_inputs = {
                k: v.to(effective_device)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            # 设置停止标记
            stop_tokens = [self.tokenizer.eos_token_id]
            if hasattr(self.tokenizer, 'eot_id'):
                stop_tokens.append(self.tokenizer.eot_id)

            # 检查是否有未识别的参数
            remaining_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('🤖', '⚙️', '💬', '🎭', '🌡️', '🎯', '📏', '🎲', '🎮', '🔄', '🖼️'))}
            if remaining_kwargs:
                print(f"[Qwen3VL_Chat] 未识别的参数已忽略: {', '.join(remaining_kwargs.keys())}")

            # 生成参数
            gen_kwargs = {
                "max_new_tokens": 最大长度,
                "do_sample": True,
                "temperature": 温度,
                "top_p": top_p,
                "eos_token_id": stop_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }

            # 生成文本
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            input_ids_len = model_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(
                outputs[0, input_ids_len:],
                skip_special_tokens=True
            )
            
            print(f"对话完成，耗时 {time.time() - start_time:.2f} 秒")
            return (text.strip(),)

        except (ValueError, RuntimeError) as e:
            error_message = f"错误: {str(e)}"
            print(error_message)
            return (error_message,)
        finally:
            if not 保持模型加载:
                self.clear_model_resources()


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Advanced": Qwen3VL_Advanced,
    "Qwen3VL_Chat": Qwen3VL_Chat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VL_Advanced": "🍭大炮-Qwen3VL@炮老师的小课堂",
    "Qwen3VL_Chat": "🍭大炮-Qwen3VL智能对话@炮老师的小课堂",
}
