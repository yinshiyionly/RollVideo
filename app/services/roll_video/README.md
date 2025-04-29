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
- 支持添加背景音乐。
- 支持从文件读取文本内容。

## 技术细节：编码策略

为了平衡质量、性能和文件格式兼容性，服务根据背景颜色 (`bg_color`) 的透明度自动选择不同的编码策略：

1.  **需要透明背景** (Alpha < 1.0 或 < 255):
    *   **目标**: 保证高质量的透明通道。
    *   **编码器**: `prores_ks` (Profile 4444)。这是一个高质量的专业编码器，良好支持Alpha通道。
    *   **处理方式**: **CPU** 进行编码 (ProRes目前没有广泛可用的GPU加速方案)。
    *   **输出格式**: `.mov`。这是支持ProRes编码和透明通道的常用容器。
    *   **性能**: 由于使用CPU和高质量编码，速度可能相对较慢，但质量最好。
    *   **FFmpeg 命令示例** (假设无音频, 宽450, 高700, 帧率30):
        ```bash
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgba -r 30 -i - \
          -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le -alpha_bits 16 -vendor ap10 \
          -map 0:v:0 \
          output.mov
        ```
        *   `-f rawvideo ... -i -`: 从标准输入读取原始 RGBA 像素数据。
        *   `-c:v prores_ks ...`: 使用 ProRes 4444 编码器进行 CPU 编码，保留 Alpha 通道。
        *   `-map 0:v:0`: 仅映射视频流。

2.  **不需要透明背景** (Alpha = 1.0 或 255):
    *   **目标**: 优先考虑编码速度，生成适合网络播放的格式。
    *   **编码器 (首选)**: `h264_nvenc`。尝试使用 **Nvidia GPU** 进行硬件加速编码，速度快。
    *   **编码器 (备选)**: `libx264`。如果 GPU 尝试失败（例如系统无兼容GPU、驱动问题、ffmpeg未启用NVENC），则自动回退到使用 **CPU** 进行编码。
    *   **输出格式**: `.mp4`。使用 `-movflags +faststart` 参数优化，适合网络流式播放。
    *   **性能**: GPU可用时速度最快；若回退到CPU，速度依然受益于直接管道传输，优于旧方案。
    *   **FFmpeg 命令示例 (GPU 优先)** (假设无音频, 宽450, 高700, 帧率30):
        ```bash
        # 尝试 GPU
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 30 -i - \
          -c:v h264_nvenc -preset p3 -rc:v vbr -cq:v 21 -b:v 0 -movflags +faststart \
          -map 0:v:0 \
          output.mp4
        ```
        *   `-f rawvideo ... -i -`: 从标准输入读取原始 RGB24 像素数据。
        *   `-c:v h264_nvenc ...`: 尝试使用 Nvidia H.264 GPU 编码器，`p3` 预设，VBR 质量 21。
        *   `-movflags +faststart`: 优化 MP4 文件结构以利于流式播放。
        *   `-map 0:v:0`: 仅映射视频流。
    *   **FFmpeg 命令示例 (CPU 回退)** (GPU失败后自动执行):
        ```bash
        # CPU 回退
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 30 -i - \
          -c:v libx264 -crf 21 -preset medium -pix_fmt yuv420p -movflags +faststart \
          -map 0:v:0 \
          output.mp4
        ```
        *   `-c:v libx264 ...`: 使用 H.264 CPU 编码器，CRF 质量 21，`medium` 预设。
        *   `-pix_fmt yuv420p`: 设置与 MP4 兼容的像素格式。

**核心优化**: 无论是哪种路径，视频帧数据都是在内存中生成后，通过**管道直接传输给 FFmpeg**，避免了因读写大量临时文件而造成的严重性能瓶颈。

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
| line_spacing | int | 20 | 行间距 |
| char_spacing | int | 0 | 字符间距 |
| fps | int | 30 | 视频帧率 |
| scroll_speed | int | 2 | 滚动速度(像素/帧) |
| audio_path | str | None | 背景音乐路径 |

## 依赖安装

```bash
# Python 库 (参考 requirements.txt)
pip install pillow numpy tqdm

# 核心依赖：FFmpeg
# 需要安装 FFmpeg。为了使用GPU加速(可选)，
# FFmpeg 需要编译时启用 NVENC 支持。
# 检查是否支持: ffmpeg -encoders | grep nvenc
```

## 注意事项

- 滚动速度越大，视频时长越短。
- 文本内容较多时，即使有优化，处理时间依然较长，尤其使用CPU编码时。
- 如果使用中文字体，请确保指定了正确的字体文件路径。
- **GPU 加速依赖**: 若需使用GPU加速（仅限不透明视频），请确保：
    - 安装了兼容的 Nvidia 显卡及最新驱动。
    - 安装了支持 NVENC 的 FFmpeg 版本。

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




