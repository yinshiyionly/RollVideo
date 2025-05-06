# 滚动视频制作服务

## 实现方案

该服务使用Python实现，主要依赖以下库：
- **Pillow (PIL)**: 用于文字渲染和图片处理。
- **NumPy**: **用于高效处理图像帧数据**。
- **FFmpeg**: 作为核心的视频编码引擎，通过`subprocess`直接调用。
- **Numba (可选)**: 用于JIT编译关键计算代码，显著提升性能。

服务通过 Pillow 将文本渲染成包含透明通道（RGBA）的长图片，然后在内存中逐帧生成视频画面，通过**管道 (Pipe)** 直接将原始像素数据流式传输给 FFmpeg 进程进行编码，避免了磁盘I/O瓶颈。

## 功能特点

- 支持自定义视频尺寸、字体、颜色（包括透明度）等参数。
- **自动处理文本换行**和排版。
- **平滑的滚动**效果。
- **自动优化编码策略**：根据背景是否透明，自动选择最优的编码器和输出格式（透明使用CPU+ProRes+MOV，不透明优先尝试GPU+H.264+MP4）。
- **GPU加速（可选）**：对于不透明视频，优先尝试使用Nvidia GPU (NVENC) 加速H.264编码，若失败则自动回退到CPU编码。
- **多线程帧生成**：利用多核CPU并行生成帧，提高渲染速度。
- **批处理帧生成**：减少线程切换开销，提高性能。
- **内存优化**：使用帧位置缓存和共享内存缓冲区，减少内存分配和复制开销。
- **JIT编译加速**：使用Numba编译关键计算路径，显著提高计算速度。
- **CPU控制**：透明视频渲染时自动限制CPU核心使用，避免资源竞争。
- **跳帧渲染**：可选择降低实际渲染帧率，然后让ffmpeg进行帧率转换，显著提高渲染速度。
- **分辨率缩放**：可选择降低渲染分辨率以提高速度，牺牲部分质量。
- 支持添加背景音乐。
- 支持从文件读取文本内容。

## 最新性能优化

为了提高渲染速度，我们实施了以下优化：

1. **多线程帧生成**：
   - 使用线程池并行生成帧数据
   - 帧队列缓冲区，预填充策略
   - 自动调整线程数，根据CPU核心数平衡编码和生成需求

2. **帧位置缓存优化**：
   - 缓存计算的滚动位置和图像切片参数
   - 减少重复计算的开销
   - 重用帧缓冲区，减少内存分配

3. **批处理帧生成**：
   - 每个线程任务一次处理多个帧
   - 减少线程调度和切换开销
   - 提高线程利用率和吞吐量

4. **共享内存优化**：
   - 使用mmap创建共享内存缓冲区
   - 在线程间传递内存引用而非整个帧数据
   - 减少大块内存拷贝，降低内存使用

5. **Numba JIT编译加速**：
   - 关键计算路径使用Numba编译为机器码
   - 自动向量化和SIMD优化
   - 图像混合计算性能提升5-100倍

6. **透明视频CPU控制**：
   - 自动限制ProRes编码的线程数
   - 设置环境变量控制各类库的线程使用
   - 可选的CPU亲和性设置（Linux）
   - 防止在多核系统上占用过多资源

7. **跳帧渲染**：
   - 实际仅渲染每X帧中的1帧
   - 由FFmpeg负责生成目标帧率的视频
   - 大幅减少需要生成的帧数量

8. **降低处理分辨率**：
   - 可设置缩放因子(0.5-1.0)降低处理分辨率
   - 对不透明视频有效
   - 同时调整字体大小和滚动速度以保持一致效果

9. **NVENC硬件优化**：
   - 针对NVIDIA GPU添加专门优化参数
   - 增加表面缓冲区数量，降低延迟
   - 禁用场景切换检测以提高编码速度

## 技术细节：编码策略

为了平衡质量、性能和文件格式兼容性，服务根据背景颜色 (`bg_color`) 的透明度自动选择不同的编码策略：

1.  **需要透明背景** (Alpha < 1.0 或 < 255):
    *   **目标**: 在追求速度的同时保持透明通道可用。
    *   **编码器**: `prores_ks` (Profile 0)。使用代理质量以提高编码速度。
    *   **处理方式**: **CPU** 进行编码 (ProRes目前没有广泛可用的GPU加速方案)。
    *   **线程控制**: 自动限制CPU使用，防止占用过多核心。
    *   **输出格式**: `.mov`。这是支持ProRes编码和透明通道的常用容器。
    *   **性能**: 使用低质量设置、多线程、批处理和Numba优化，速度有显著提升。
    *   **FFmpeg 命令示例** (假设无音频, 宽450, 高700, 帧率30):
        ```bash
        ffmpeg -y \
          -f rawvideo -vcodec rawvideo -s 450x700 -pix_fmt rgba -r 30 -i - \
          -c:v prores_ks -profile:v 0 -pix_fmt yuva444p -alpha_bits 8 -vendor ap10 \
          -threads 7 -map 0:v:0 \
          output.mov
        ```
        *   `-f rawvideo ... -i -`: 从标准输入读取原始 RGBA 像素数据。
        *   `-c:v prores_ks ...`: 使用 ProRes 编码器进行 CPU 编码，profile 0 (代理质量)。
        *   `-threads 7`: 限制编码使用的CPU线程数。
        *   `-map 0:v:0`: 仅映射视频流。

2.  **不需要透明背景** (Alpha = 1.0 或 255):
    *   **目标**: 最大化编码速度，牺牲部分质量。
    *   **编码器 (首选)**: `h264_nvenc`。使用 **Nvidia GPU** 进行硬件加速编码，速度快。
    *   **编码器 (备选)**: `libx264`。使用 **CPU** 的ultrafast预设进行编码。
    *   **输出格式**: `.mp4`。使用 `-movflags +faststart` 参数优化，适合网络流式播放。
    *   **性能**: 使用多线程生成帧 + GPU加速 + 跳帧 + 批处理 + Numba优化，速度大幅提升。
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

**核心优化**: 多线程并行帧生成 + 批处理模式 + 帧缓存 + 共享内存 + Numba JIT编译 → 在保证可用的画质下显著提升速度。

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
| fps | int | 24 | 视频帧率 |
| scroll_speed | int | 3 | 滚动速度(像素/帧) |
| audio_path | str | None | 背景音乐路径 |
| scale_factor | float | 1.0 | 缩放因子(0.5-1.0)，小于1时降低处理分辨率以提高速度 |
| frame_skip | int | 1 | 跳帧率，大于1时减少渲染帧以提高速度 (例如2表示只渲染一半的帧) |

## 性能建议

根据您的硬件配置（8核CPU、30G内存、10G显存），以下是推荐设置：

- **适用于最大速度(8核CPU)**:
  - `scale_factor=0.75`：降低25%分辨率
  - Numba JIT编译优化：确保安装numba库
  - 批处理帧生成：自动以4帧为单位批量处理
  - 预期速度提升：3-4倍，同时控制CPU使用

- **适用于平衡速度/质量**:
  - `scale_factor=0.9`：轻微降低分辨率
  - `frame_skip=1`：保持所有帧
  - 多线程和批处理优化：自动使用
  - 预期速度提升：2-2.5倍

- **透明视频最佳设置**:
  - CPU线程控制：自动限制到合理数量
  - 批处理优化和帧缓存：自动应用
  - Numba JIT编译：显著加速alpha混合计算
  - 预期速度提升：2-3倍

- **超大文本处理建议**:
  - `scale_factor=0.6`
  - 使用全部优化（帧缓存、批处理、共享内存、JIT编译）
  - 内存使用会降低到原来的25-30%
  - CPU使用更平衡

## 依赖安装

```bash
# Python 库 (参考 requirements.txt)
pip install pillow numpy tqdm concurrent-futures-extra mmap psutil

# 可选但强烈推荐：安装Numba以获得显著性能提升
pip install numba

# 核心依赖：FFmpeg
# 需要安装 FFmpeg。为了使用GPU加速(可选)，
# FFmpeg 需要编译时启用 NVENC 支持。
# 检查是否支持: ffmpeg -encoders | grep nvenc
```

## 注意事项

- 滚动速度越大，视频时长越短。
- 较低的frame_skip值 (1-2) 保持较好的滚动平滑度。
- scale_factor不应低于0.5，否则文本可能难以辨认。
- **滚动速度与帧率关系**：
  - 当滚动速度 > 4 像素/帧时，系统会自动提高帧率以保持流畅度
  - 建议搭配：低速滚动(1-3像素/帧)使用24fps，高速滚动(5+像素/帧)使用30fps
  - 视觉流畅度与性能的平衡点：24fps×3像素/帧 或 30fps×2像素/帧
- **GPU 加速依赖**: 若需使用GPU加速（仅限不透明视频），请确保：
    - 安装了兼容的 Nvidia 显卡及最新驱动。
    - 安装了支持 NVENC 的 FFmpeg 版本。
- 透明视频使用的ProRes编码自动限制CPU使用，不会占满所有核心。
- **Numba优化**: 首次运行时，Numba会编译关键函数，可能需要几秒钟。后续运行将使用缓存的编译结果，速度大幅提升。

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

# 滚动视频生成器

本模块提供了两种不同的方法来生成滚动视频（类似片尾字幕效果）：

1. 传统方法（逐帧裁剪）
2. FFmpeg滤镜方法（使用crop滤镜和时间表达式）

## 传统方法 vs. FFmpeg滤镜方法

### 传统方法（逐帧裁剪）

传统方法通过逐帧处理实现滚动效果：

1. 渲染完整的文本图像
2. 对于每一帧，裁剪出当前视图中应该可见的部分
3. 将裁剪后的图像传递给FFmpeg编码器

**优点：**
- 支持复杂的动画和效果（可以在每帧上添加额外处理）
- 全面的控制每个帧的内容
- 跨平台兼容性好

**缺点：**
- 计算密集，需要在内存中生成所有帧
- 慢速处理大型文本（每一帧都需要单独处理）
- 高内存使用量

### FFmpeg滤镜方法（crop滤镜）

FFmpeg滤镜方法使用FFmpeg内置的crop滤镜和时间表达式实现滚动效果：

1. 渲染完整的文本图像并保存为临时文件
2. 使用FFmpeg的crop滤镜和时间表达式来模拟滚动效果
3. 由FFmpeg直接生成最终视频

**优点：**
- 显著更快的处理速度（3-7倍性能提升）
- 大幅降低CPU和内存使用率
- 支持平滑滚动（亚像素精度）
- 更简洁的代码

**缺点：**
- 受限于FFmpeg滤镜功能（不支持特别复杂的动画）
- 需要临时保存图像文件到磁盘（额外的I/O操作）

## 性能对比

基于相同的输入参数（短文本约200字符）的测试结果：

| 方法 | 场景 | 总耗时 | 帧率 | 优化比例 |
|------|------|--------|------|----------|
| 传统方法 | 不透明背景 | 3.22秒 | 193.13帧/秒 | 基准 |
| FFmpeg滤镜 | 不透明背景 | 0.93秒 | 707.3帧/秒 | 3.5倍提速 |
| FFmpeg滤镜 | 透明背景 | 2.91秒 | 126.3帧/秒 | 1.1倍提速 |

## 使用方法

### 使用传统方法

```python
from app.services.roll_video.roll_video_service import RollVideoService

service = RollVideoService()
result = service.create_roll_video(
    text="要滚动的文本内容",
    output_path="output.mp4",
    width=720,
    height=1280,
    font_path="方正黑体简体.ttf",
    font_size=24,
    font_color=[0,0,0],
    bg_color=[255,255,255,1.0],  # 不透明白色背景
    line_spacing=5,
    char_spacing=0,
    fps=30,
    scroll_speed=1,
)
```

### 使用FFmpeg滤镜方法

```python
from app.services.roll_video.roll_video_service import RollVideoService

service = RollVideoService()
result = service.create_roll_video_ffmpeg(
    text="要滚动的文本内容",
    output_path="output.mp4",
    width=720,
    height=1280,
    font_path="方正黑体简体.ttf",
    font_size=24,
    font_color=[0,0,0],
    bg_color=[255,255,255,1.0],  # 不透明白色背景
    line_spacing=5,
    char_spacing=0,
    fps=30,
    scroll_speed=1,
)
```

## 结论

对于大多数常见用例，特别是大型文本生成，建议使用 **FFmpeg滤镜方法**，它能够提供显著的性能优势。对于需要在每个帧上应用复杂效果的特殊用例，可以继续使用传统方法。