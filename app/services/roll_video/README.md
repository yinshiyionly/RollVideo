# 滚动视频制作服务

## 实现方案

该服务使用Python实现，主要依赖以下库：
- **Pillow (PIL)**: 用于文字渲染和图片处理。
- **NumPy**: **用于高效处理图像帧数据**。
- **FFmpeg**: 作为核心的视频编码引擎，通过`subprocess`直接调用。
- **concurrent.futures**: 支持多线程并行处理帧生成。

服务通过 Pillow 将文本渲染成包含透明通道（RGBA）的长图片，然后在内存中**并行生成视频帧**，通过**生产者-消费者模式和管道 (Pipe)** 直接将原始像素数据流式传输给 FFmpeg 进程进行编码，避免了磁盘I/O瓶颈。

## 功能特点

- 支持自定义视频尺寸、字体、颜色（包括透明度）等参数。
- **自动处理文本换行**和排版。
- **平滑的滚动**效果。
- **自动优化编码策略**：根据背景是否透明，自动选择最优的编码器和输出格式（透明使用CPU+ProRes+MOV，不透明优先尝试GPU+H.265/HEVC或H.264+MP4）。
- **多级GPU加速**：对于不透明视频，优先尝试使用Nvidia GPU的HEVC编码器，其次是H.264 (NVENC)，最后才回退到CPU编码。
- **并行处理优化**：使用多线程并行生成视频帧，大幅提升处理速度。
- **生产者-消费者模式**：一边生成帧，一边发送给ffmpeg，充分利用CPU多核心。
- 支持添加背景音乐。
- 支持从文件读取文本内容。

## 技术细节：多重优化机制

服务采用了多重优化机制来提高编码速度：

1. **并行帧生成**：
   - 使用`ThreadPoolExecutor`并行生成视频帧
   - 可配置的工作线程数（默认根据CPU核心数自动调整）
   - 线程数上限为8以避免过度竞争资源

2. **生产者-消费者架构**：
   - 主线程（生产者）负责生成帧并放入缓冲区
   - 消费者线程负责从缓冲区取出帧并写入ffmpeg管道
   - 可配置的帧缓冲区大小（默认为帧率的80%）

3. **优化的编码器选择**：
   - 透明视频：`prores_ks` Profile 4444（保持高质量透明通道）
   - 不透明视频分三级：
     1. 优先尝试：`hevc_nvenc`（H.265/HEVC，提供更高压缩率）
     2. 次优先尝试：`h264_nvenc`（H.264 GPU加速）
     3. 最后回退：`libx264`（CPU编码，使用"fast"预设提升速度）

4. **内存优化**：
   - 预计算所有帧的滚动位置
   - 优化Alpha通道混合计算，使用矢量化操作
   - 减少不必要的数据复制操作

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
    *   **编码器 (第一选择)**: `hevc_nvenc`。尝试使用 **Nvidia GPU** 进行H.265/HEVC硬件加速编码，更好的压缩率和速度。
    *   **编码器 (第二选择)**: `h264_nvenc`。如果HEVC尝试失败，使用H.264 **GPU**加速编码，速度仍然较快。
    *   **编码器 (最后选择)**: `libx264`。如果 GPU 尝试全部失败，则自动回退到使用 **CPU** 进行编码，使用"fast"预设提高速度。
    *   **输出格式**: `.mp4`。使用 `-movflags +faststart` 参数优化，适合网络流式播放。
    *   **性能**: 多级回退保证在各种环境下都能获得最佳性能。
    *   **FFmpeg 命令示例 (HEVC GPU 优先)** (假设无音频, 宽450, 高700, 帧率30):
        ```bash
        # 尝试 HEVC/H.265 GPU
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 30 -i - \
          -c:v hevc_nvenc -preset p3 -rc:v vbr -cq:v 24 -b:v 0 -pix_fmt yuv420p -tag:v hvc1 -movflags +faststart \
          -map 0:v:0 \
          output.mp4
        ```
        *   `-c:v hevc_nvenc ...`: 使用 Nvidia HEVC GPU 编码器，`p3` 预设，VBR 质量 24。
        *   `-tag:v hvc1`: 添加兼容标签以提高兼容性。
    *   **FFmpeg 命令示例 (H.264 GPU 回退)** (HEVC失败后自动执行):
        ```bash
        # 尝试 H.264 GPU
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 30 -i - \
          -c:v h264_nvenc -preset p3 -rc:v vbr -cq:v 21 -b:v 0 -pix_fmt yuv420p -movflags +faststart \
          -map 0:v:0 \
          output.mp4
        ```
    *   **FFmpeg 命令示例 (CPU 回退)** (GPU都失败后自动执行):
        ```bash
        # CPU 回退
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgb24 -r 30 -i - \
          -c:v libx264 -crf 23 -preset fast -tune fastdecode -pix_fmt yuv420p -movflags +faststart \
          -map 0:v:0 \
          output.mp4
        ```
        *   `-preset fast`: 使用更快的预设而不是默认的`medium`。
        *   `-tune fastdecode`: 优化解码速度，减少解码资源需求。

**核心优化**: 通过并行帧生成和生产者-消费者模式，结合管道直接传输，实现了最大化的渲染和编码效率。

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
| worker_threads | int | None | 用于帧处理的工作线程数 (默认为CPU核心数，最大8) |
| frame_buffer_size | int | None | 帧缓冲区大小 (默认为fps的80%，最大24) |

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
- 文本内容较多时处理速度已经过优化，但仍需考虑内容量。
- 如果使用中文字体，请确保指定了正确的字体文件路径。
- **GPU 加速依赖**: 若需使用GPU加速（仅限不透明视频），请确保：
    - 安装了兼容的 Nvidia 显卡及最新驱动。
    - 安装了支持 NVENC 的 FFmpeg 版本。
    - 使用`hevc_nvenc`需要更高版本的GPU和驱动支持。

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




