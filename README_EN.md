# ComfyUI-Qwen3VL-DP

🍭 **DaPao-Qwen3VL** ComfyUI Custom Node Collection - Integrating Alibaba Cloud Qwen Team's Qwen3-VL Multimodal Large Language Model Series.

ComfyUI custom node integrating the Qwen3-VL multimodal large language model series developed by Qwen team, Alibaba Cloud.

## Features

### 🎯 Core Capabilities

- **🖼️ Image Understanding**: Detailed analysis and description of image content
- **🎥 Video Understanding**: Process video frame sequences to understand video content
- **🧠 Multimodal Reasoning**: Reasoning combining visual and text information
- **✨ Prompt Generation**: Generate optimized prompts for text-to-image AI
- **📁 Batch Processing**: Batch tagging and comparison analysis features

### 🍧 Feature Showcase
- **🤖 Simple Reverse Prompting [Single Image/Multiple Images/Video]**
![alt text](功能展示/01单图反推.png)
![alt text](功能展示/02多图对比.png)
![alt text](功能展示/03视频反推.png)
- **🤖 Intelligent Dialogue [With Image Analysis]**
![alt text](功能展示/智能对话.png)
- **🤖 Batch Folder Tagging**
![alt text](功能展示/批量打标.png)
- **🤖 Folder Comparison Tagging [Suitable for kontext/Qwen-edit]**
![alt text](功能展示/文件夹对比打标.png)

### 🚀 Key Enhancements

- **Visual Agent**: Recognizes UI elements, understands functions, completes tasks
- **Visual Coding Boost**: Generates Draw.io/HTML/CSS/JS code from images/videos
- **Advanced Spatial Perception**: Judges object positions, viewpoints, and occlusions
- **Long Context Understanding**: Native 256K context support, expandable to 1M
- **Enhanced Multimodal Reasoning**: Excels in STEM/Math domains
- **Upgraded Visual Recognition**: Recognizes celebrities, anime, products, landmarks, flora/fauna, etc.
- **Expanded OCR**: Supports 32 languages, robust in low light, blur, and tilt

### 💡 Model Architecture Updates

1. **Interleaved-MRoPE**: Full-frequency allocation over time, width, and height via robust positional embeddings
2. **DeepStack**: Fuses multi-level ViT features to capture fine-grained details
3. **Text-Timestamp Alignment**: Precise timestamp-grounded event localization

## Supported Models

### Qwen3-VL Series

- **Qwen3-VL-2B-Instruct** / **Qwen3-VL-2B-Thinking**
  - VRAM: 4GB (FP16) / 2.5GB (8-bit) / 1.5GB (4-bit)
  - FP8 version: 2.5GB

- **Qwen3-VL-4B-Instruct** / **Qwen3-VL-4B-Thinking** (Default)
  - VRAM: 6GB (FP16) / 3.5GB (8-bit) / 2GB (4-bit)
  - FP8 version: 2.5GB

- **Qwen3-VL-8B-Instruct** / **Qwen3-VL-8B-Thinking**
  - VRAM: 12GB (FP16) / 7GB (8-bit) / 4.5GB (4-bit)
  - FP8 version: 7.5GB

- **Qwen3-VL-32B-Instruct** / **Qwen3-VL-32B-Thinking**
  - VRAM: 28GB (FP16) / 14GB (8-bit) / 8.5GB (4-bit)
  - FP8 version: 24GB

### Qwen2.5-VL Series (Backward Compatible)

- **Qwen2.5-VL-3B-Instruct**: 6GB / 3.5GB / 2GB
- **Qwen2.5-VL-7B-Instruct**: 15GB / 8.5GB / 5GB

### Community Models

- **Huihui-Qwen3-VL-8B-Instruct-Abliterated**: 12GB / 7GB / 4.5GB
  - ⚠️ **Warning**: This model has safety filtering removed and may generate sensitive content
  - For research and testing environments only
  - Based on Qwen3-VL-8B abliterated version

## 📦 Installation Guide

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-Qwen3VL-DP.git
cd ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```

### Method 2: Manual Download

1. Download project files to `ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP/`
2. Install dependencies:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Qwen3VL-DP
pip install -r requirements.txt
```
![alt text](功能展示/自动下载模型.png)

### Optional: Flash Attention 2 (Highly Recommended)

If you have a compatible NVIDIA GPU (Compute Capability >= 8.0, such as RTX 30/40 series):

```bash
pip install flash-attn --no-build-isolation
```

### Restart ComfyUI

Restart ComfyUI to load the new nodes.

## 🗂️ Model Path and Configuration

### Automatic Download

On first use, models will be automatically downloaded from Hugging Face to the ComfyUI model directory:

```
ComfyUI/models/prompt_generator/
```

**Workflow:**
1. 🔍 **Check Model**: Automatically checks if the model exists when running the node
2. 📥 **Auto Download**: If the model doesn't exist, automatically downloads from Hugging Face
3. ✅ **Direct Use**: If the model exists, loads it directly without re-downloading

**Example Paths:**
- Qwen3-VL-4B-Instruct: `ComfyUI/models/prompt_generator/Qwen3-VL-4B-Instruct/`
- Qwen3-VL-8B-Instruct: `ComfyUI/models/prompt_generator/Qwen3-VL-8B-Instruct/`

### Model Storage Requirements

| Model | Full Model | Quantized Version |
|------|----------|----------|
| Qwen3-VL-2B | ~4GB | ~2GB |
| Qwen3-VL-4B | ~8GB | ~4GB |
| Qwen3-VL-8B | ~16GB | ~8GB |
| Qwen3-VL-32B | ~64GB | ~32GB |

## Usage

### Node Types

#### 1. Qwen3-VL (Simple)

Simplified node for quick use:

**Input Parameters:**
- `model_name`: Select model
- `quantization`: Quantization level (4-bit / 8-bit / None)
- `preset_prompt`: Preset prompt template
- `custom_prompt`: Custom prompt (optional, overrides preset)
- `max_tokens`: Maximum tokens to generate
- `keep_model_loaded`: Whether to keep model loaded
- `seed`: Random seed
- `image`: Input image (optional)
- `video`: Input video frame sequence (optional)

#### 2. Qwen3-VL (Advanced)

Advanced node with more control parameters:

**Additional Parameters:**
- `temperature`: Sampling temperature (0.1-1.0)
- `top_p`: Nucleus sampling parameter (0.0-1.0)
- `num_beams`: Number of beams for beam search
- `repetition_penalty`: Repetition penalty
- `frame_count`: Video frame sampling count
- `device`: Device selection (auto/cuda/cpu/mps)

#### 3. Qwen3-VL (Batch Caption) ⭐ New Feature

Batch process images in a folder and automatically generate caption files:

**Core Features:**
- 📁 Batch scan all images in folder
- 📝 Auto-generate corresponding .txt caption files
- 🔄 Support file renaming and numbering
- ⏭️ Auto-skip already processed images
- 📊 Real-time progress display

**Use Cases:**
- Batch annotation for training datasets
- Bulk caption generation for image libraries
- Batch generation of AI art prompts

**Detailed Guide:** See [批量打标使用说明.md](批量打标使用说明.md) (Chinese)

### Preset Prompt Templates

1. **Prompt Style - Tags**: Generate comma-separated tag list
2. **Prompt Style - Simple**: Single sentence concise description
3. **Prompt Style - Detailed**: 2-3 sentence detailed description
4. **Prompt Style - Extreme Detailed**: Highly detailed paragraph description
5. **Prompt Style - Cinematic**: Evocative cinematic style description
6. **Creative - Detailed Analysis**: Detailed analysis by sections
7. **Creative - Summarize Video**: Summarize key video events
8. **Creative - Short Story**: Create imaginative story
9. **Creative - Refine & Expand Prompt**: Optimize and enhance existing prompts

### Workflow Examples

#### Image Description

```
[Load Image] -> [Qwen3-VL (Simple)] -> [Save Text]
                      ↓
                preset_prompt: "Prompt Style - Detailed"
```

#### Video Analysis

```
[Load Video] -> [Video Frames] -> [Qwen3-VL (Advanced)] -> [Save Text]
                                        ↓
                                  frame_count: 16
                                  preset_prompt: "Creative - Summarize Video"
```

#### Prompt Optimization

```
[Load Image] -> [Qwen3-VL (Simple)] -> [Qwen3-VL (Simple)] -> [Text to Image]
                      ↓                      ↓
                "Prompt Style - Tags"    "Creative - Refine & Expand Prompt"
```

## Quantization Notes

- **None (FP16)**: Best quality, highest VRAM requirement
- **8-bit**: Balanced quality and VRAM usage
- **4-bit**: Lowest VRAM requirement, slightly reduced quality
- **FP8 Models**: Pre-quantized models, require GPU with Compute Capability >= 8.9 (e.g., RTX 4090)

The node automatically detects available VRAM and reduces quantization level if necessary.

## Custom Models

You can add custom models by creating a `custom_models.json` file:

```json
{
    "hf_models": {
        "My-Custom-Model": {
            "repo_id": "username/model-name",
            "default": false,
            "quantized": false,
            "vram_requirement": {
                "full": 8.0,
                "8bit": 4.5,
                "4bit": 2.5
            }
        }
    }
}
```

## Performance Optimization Tips

1. **Use Flash Attention 2**: Significantly improves speed and reduces VRAM usage
2. **Choose Appropriate Model Size**: Select model based on your hardware
3. **Enable Quantization**: Use 4-bit or 8-bit quantization when VRAM is limited
4. **Keep Model Loaded**: Avoid overhead of repeatedly loading models
5. **Adjust Frame Count**: Reduce `frame_count` for video processing to save VRAM

## System Requirements

### Minimum Requirements
- Python 3.8+
- PyTorch 2.0+
- 8GB System Memory
- 2GB VRAM (2B model with 4-bit quantization)

### Recommended Configuration
- Python 3.10+
- PyTorch 2.1+
- 16GB+ System Memory
- 8GB+ VRAM (NVIDIA GPU)
- CUDA 11.8+ (for GPU acceleration)

### Best Experience
- 24GB+ VRAM (RTX 3090/4090)
- Flash Attention 2 support
- NVMe SSD (for model loading)

## Troubleshooting

### Out of Memory

- Use smaller model (2B or 4B)
- Enable 4-bit quantization
- Reduce `max_tokens` and `frame_count`
- Close other VRAM-consuming applications

### Model Loading Failed

- Check network connection
- Ensure sufficient disk space
- Verify Hugging Face access permissions

### Poor Generation Quality

- Try larger model
- Adjust `temperature` and `top_p`
- Use more detailed prompts
- Reduce `repetition_penalty`

## License

- **Qwen3-VL Models**: Apache-2.0 License
- **Qwen2.5-VL Models**: Apache-2.0 License
- **This Integration Code**: MIT License

## Acknowledgments

- Qwen Team, Alibaba Cloud - Developing Qwen3-VL models
- ComfyUI - Providing powerful node-based UI framework
- Hugging Face - Model hosting and Transformers library

## Contributing

Standing on the shoulders of giants! This project references the following open-source nodes:
- https://github.com/1038lab/ComfyUI-QwenVL?tab=readme-ov-file
- https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two

## Changelog

### v1.2.0 (2025-11-14)

- ✨ **New Batch Caption Node**: Support batch processing of images in folders
- 📁 Support auto-scanning and processing multiple image formats
- 🔄 Support file renaming and auto-numbering
- 📝 Support adding prefix and suffix text
- ⏭️ Smart skip already processed images
- 📊 Real-time progress display and statistics
- 📖 Added detailed batch caption documentation

### v1.1.0 (2025-11-11)

- ✨ Added community model support: Huihui-Qwen3-VL-8B-Instruct-Abliterated
- 🔧 Optimized code comments with detailed docstrings
- ⚠️ Added abliterated model warning mechanism
- 🐛 Fixed missing trust_remote_code parameter
- 📝 Improved code readability and maintainability

### v1.0.0 (2025-11-11)

- Initial release
- Support for all Qwen3-VL series models
- Image and video understanding support
- Simple and Advanced node variants
- Automatic VRAM management and quantization
- Preset prompt templates
- Custom model support

## Contributing

Feedback and suggestions are welcome!
