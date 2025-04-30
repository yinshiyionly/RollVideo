# 滚动视频制作服务

## 实现方案

该服务使用Python实现，主要依赖以下库：
- **Pillow (PIL)**: 用于文字渲染和图片处理。
- **NumPy**: **用于高效处理图像帧数据**。
- **FFmpeg**: 作为核心的视频编码引擎，通过`subprocess`直接调用。

服务通过 Pillow 将文本渲染成包含透明通道（RGBA）的长图片，然后在内存中逐帧生成视频画面，通过**管道 (Pipe)** 直接将原始像素数据流式传输给 FFmpeg 进程进行编码，避免了磁盘I/O瓶颈。

## 功能特点

- 支持自定义视频尺寸、字体、颜色（包括透明度）等参数。
- **自动处理文本换行**和排版。
- **平滑的滚动**效果。
- **自动优化编码策略**：根据背景是否透明，自动选择最优的编码器和输出格式（透明使用CPU+ProRes+MOV，不透明优先尝试GPU+H.264+MP4）。
- **GPU加速（可选）**：对于不透明视频，优先尝试使用Nvidia GPU (NVENC) 加速H.264编码，若失败则自动回退到CPU编码。
- **多线程帧生成**：利用多核CPU并行生成帧，提高渲染速度。
- **跳帧渲染**：可选择降低实际渲染帧率，然后让ffmpeg进行帧率转换，显著提高渲染速度。
- **分辨率缩放**：可选择降低渲染分辨率以提高速度，牺牲部分质量。
- 支持添加背景音乐。
- 支持从文件读取文本内容。

## 最新性能优化

为了提高渲染速度，我们实施了以下优化：

1. **多线程帧生成**：
   - 使用线程池并行生成帧数据
   - 帧队列缓冲区，预填充策略
   - 默认使用4线程（可根据CPU核心数自动调整）

2. **编码质量调整**：
   - 透明视频：ProRes配置从高质量(4)调整为代理质量(0)，位深度降低
   - 不透明视频：NVENC使用更快的preset(p4)，质量值从21调整到28
   - CPU回退：使用ultrafast预设，添加fastdecode优化

3. **跳帧渲染**：
   - 实际仅渲染每X帧中的1帧
   - 由FFmpeg负责生成目标帧率的视频
   - 大幅减少需要生成的帧数量

4. **降低处理分辨率**：
   - 可设置缩放因子(0.5-1.0)降低处理分辨率
   - 对不透明视频有效
   - 同时调整字体大小和滚动速度以保持一致效果

5. **NVENC硬件优化**：
   - 针对NVIDIA GPU添加专门优化参数
   - 增加表面缓冲区数量，降低延迟
   - 禁用场景切换检测以提高编码速度

## 技术细节：编码策略

为了平衡质量、性能和文件格式兼容性，服务根据背景颜色 (`bg_color`) 的透明度自动选择不同的编码策略：

1.  **需要透明背景** (Alpha < 1.0 或 < 255):
    *   **目标**: 在追求速度的同时保持透明通道可用。
    *   **编码器**: `prores_ks` (Profile 0)。使用代理质量以提高编码速度。
    *   **处理方式**: **CPU** 进行编码 (ProRes目前没有广泛可用的GPU加速方案)。
    *   **输出格式**: `.mov`。这是支持ProRes编码和透明通道的常用容器。
    *   **性能**: 使用低质量设置和多线程优化，速度有显著提升。
    *   **FFmpeg 命令示例** (假设无音频, 宽450, 高700, 帧率30):
        ```bash
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgba -r 30 -i - \
          -c:v prores_ks -profile:v 0 -pix_fmt yuva444p -alpha_bits 8 -vendor ap10 \
          -map 0:v:0 \
          output.mov
        ```
        *   `-f rawvideo ... -i -`: 从标准输入读取原始 RGBA 像素数据。
        *   `-c:v prores_ks ...`: 使用 ProRes 编码器进行 CPU 编码，profile 0 (代理质量)。
        *   `-map 0:v:0`: 仅映射视频流。

2.  **不需要透明背景** (Alpha = 1.0 或 255):
    *   **目标**: 最大化编码速度，牺牲部分质量。
    *   **编码器 (首选)**: `h264_nvenc`。使用 **Nvidia GPU** 进行硬件加速编码，速度快。
    *   **编码器 (备选)**: `libx264`。使用 **CPU** 的ultrafast预设进行编码。
    *   **输出格式**: `.mp4`。使用 `-movflags +faststart` 参数优化，适合网络流式播放。
    *   **性能**: 使用多线程生成帧 + GPU加速 + 跳帧，速度大幅提升。
    *   **FFmpeg 命令示例 (GPU)** (假设无音频, 宽450, 高700, 跳帧, 实际渲染15fps输出30fps):
        ```bash
        # GPU 编码
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 15 -i - \
          -c:v h264_nvenc -preset p4 -rc:v vbr -cq:v 28 -b:v 0 -pix_fmt yuv420p \
          -gpu 0 -surfaces 64 -delay 0 -no-scenecut 1 -movflags +faststart \
          -r 30 -map 0:v:0 \
          output.mp4
        ```
        *   `-f rawvideo ... -r 15 -i -`: 从标准输入读取原始 RGB24 像素数据，低于目标帧率。
        *   `-c:v h264_nvenc ...`: 使用 Nvidia H.264 GPU 编码器，`p4` 预设，更高速度。
        *   `-gpu 0 -surfaces 64...`: NVENC专用优化参数。
        *   `-r 30`: 输出目标帧率，ffmpeg会自动进行帧率转换。
    *   **FFmpeg 命令示例 (CPU 回退)**:
        ```bash
        # CPU 回退
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 15 -i - \
          -c:v libx264 -crf 28 -preset ultrafast -tune fastdecode -pix_fmt yuv420p -movflags +faststart \
          -r 30 -map 0:v:0 \
          output.mp4
        ```
        *   `-c:v libx264 ...`: 使用 H.264 CPU 编码器，CRF 质量 28，`ultrafast` 预设。
        *   `-tune fastdecode`: 优化解码速度，进一步提高性能。

**核心优化**: 多线程并行帧生成 + 跳帧渲染 + 降低分辨率 + 降低质量 → 在保证可用的画质下显著提升速度。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | str | 必填 | 要展示的文本内容 |
| output_path | str | 必填 | **期望的**输出视频路径 (最终扩展名会根据透明度自动调整为`.mov`或`.mp4`) |
| width | int | 1080 | 视频宽度 |
| height | int | 1920 | 视频高度 |
| font_path | str | 系统默认 | 字体文件路径 |
| font_size | int | 40 | 字体大小 |
| font_color | (r,g,b) | (255,255,255) | 字体颜色 |
| bg_color | (r,g,b) 或 (r,g,b,a) | (0,0,0,255) | 背景颜色 (元组)。Alpha值(0-255或0.0-1.0)决定是否透明。省略alpha或alpha=255/1.0表示不透明。 |
| line_spacing | float | 0.5 | 行间距比例因子 (例如 0.5 表示行间距为字体大小的一半) |
| char_spacing | int | 0 | 字符间距 |
| fps | int | 30 | 视频帧率 |
| scroll_speed | int | 2 | 滚动速度(像素/帧) |
| audio_path | str | None | 背景音乐路径 |
| scale_factor | float | 1.0 | 缩放因子(0.5-1.0)，小于1时降低处理分辨率以提高速度 |
| frame_skip | int | 1 | 跳帧率，大于1时减少渲染帧以提高速度 (例如2表示只渲染一半的帧) |

## 性能建议

根据您的硬件配置（8核CPU、30G内存、10G显存），以下是推荐设置：

- **适用于最大速度**:
  - `scale_factor=0.75`：降低25%分辨率
  - `frame_skip=2`：只渲染一半的帧
  - 预期速度提升：2-3倍

- **适用于平衡速度/质量**:
  - `scale_factor=0.9`：轻微降低分辨率
  - `frame_skip=1`：保持所有帧
  - 预期速度提升：1.3-1.5倍

- **透明视频最佳设置**:
  - 不使用`scale_factor`（透明视频不支持）
  - `frame_skip=2`：只渲染一半的帧
  - 预期速度提升：1.5-2倍

- **超大文本处理建议**:
  - `scale_factor=0.6`
  - `frame_skip=3`
  - 减小字体大小
  - 内存使用会降低到原来的30-40%

## 依赖安装

```bash
# Python 库 (参考 requirements.txt)
pip install pillow numpy tqdm concurrent-futures-extra

# 核心依赖：FFmpeg
# 需要安装 FFmpeg。为了使用GPU加速(可选)，
# FFmpeg 需要编译时启用 NVENC 支持。
# 检查是否支持: ffmpeg -encoders | grep nvenc
```

## 注意事项

- 滚动速度越大，视频时长越短。
- 较低的frame_skip值 (1-2) 保持较好的滚动平滑度。
- scale_factor不应低于0.5，否则文本可能难以辨认。
- **GPU 加速依赖**: 若需使用GPU加速（仅限不透明视频），请确保：
    - 安装了兼容的 Nvidia 显卡及最新驱动。
    - 安装了支持 NVENC 的 FFmpeg 版本。
- 调整缩放因子时请测试效果，过低的缩放因子会影响文本清晰度。

参数：
    文字：最多5000字
    视频宽：450 px
    视频高：700 px
    滚动速度：1
    字体：PingFang
    字号：16 px
    字体颜色：#000000
    字间距：1
    行间距：1.2
    视频背景颜色：#FFFFFF