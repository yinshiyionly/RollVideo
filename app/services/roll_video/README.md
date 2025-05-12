# 滚动视频制作服务

## 实现方案

该服务使用Python实现，主要依赖以下库：
- **Pillow (PIL)**: 用于文字渲染和图片处理。
- **NumPy**: 用于高效处理图像帧数据。
- **FFmpeg**: 作为核心的视频编码引擎，通过`subprocess`直接调用。
- **Requests**: 用于获取远程背景图片。

服务通过Pillow将文本渲染成包含所需属性（包括可选的透明通道RGBA）的长图片，然后使用FFmpeg的高级滤镜功能或CUDA加速方式生成滚动效果的视频。

## 功能特点

- 支持自定义视频尺寸、字体、颜色（包括透明度）等参数。
- **自动处理文本换行**和排版。
- **支持自定义边距**：上、下、左、右边距可分别设置。
- **背景图片支持**：可设置背景图片URL，支持拉伸(stretch)和平铺(tile)两种缩放模式。
- **平滑的滚动**效果：支持精确控制滚动速度。
- **自动优化编码策略**：根据背景是否透明，自动选择最优的编码器和输出格式。
- **GPU加速**：对于不透明视频，优先尝试使用Nvidia GPU (NVENC) 加速H.264编码。
- 支持添加背景音乐。

## 滚动视频生成方法

服务提供两种不同的方法来生成滚动视频（类似片尾字幕效果）：

1. **Crop滤镜方法**：使用FFmpeg的crop滤镜和时间表达式实现滚动效果。
2. **Overlay CUDA方法**：使用FFmpeg的overlay滤镜和CUDA加速实现滚动效果。

### Crop滤镜方法

该方法使用FFmpeg内置的crop滤镜和时间表达式实现滚动效果：

1. 渲染完整的文本图像并保存为临时文件
2. 使用FFmpeg的crop滤镜和时间表达式来模拟滚动效果
3. 由FFmpeg直接生成最终视频

**优点：**
- 高效的处理速度
- 低CPU和内存使用率
- 支持平滑滚动（亚像素精度）
- 支持背景图片（拉伸或平铺模式）
- 支持自定义边距

### Overlay CUDA方法

该方法使用FFmpeg的overlay滤镜和CUDA加速功能：

1. 渲染完整的文本图像并保存为临时文件
2. 使用FFmpeg的overlay滤镜和复杂的滤镜图实现滚动效果
3. 可利用CUDA加速处理

**优点：**
- 可利用GPU加速
- 支持背景图片（拉伸或平铺模式）
- 支持自定义边距
- 适合高分辨率视频处理

## 编码策略

根据背景颜色的透明度自动选择不同的编码策略：

1. **需要透明背景** (Alpha < 255):
   - **编码器**: `prores_ks` (Profile 4444)
   - **处理方式**: **CPU** 进行编码
   - **输出格式**: `.mov`

2. **不需要透明背景** (Alpha = 255):
   - **编码器 (首选)**: `h264_nvenc`，使用 **Nvidia GPU** 进行硬件加速编码
   - **编码器 (备选)**: `libx264`，使用 **CPU** 进行编码
   - **输出格式**: `.mp4`，使用 `-movflags +faststart` 参数优化，适合网络流式播放

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
| bg_color | (r,g,b) 或 (r,g,b,a) | (0,0,0,255) | 背景颜色。Alpha值决定是否透明。省略alpha或alpha=255表示不透明。 |
| background_url | str | None | 背景图片URL，如果提供则覆盖bg_color |
| scale_mode | str | "stretch" | 背景图缩放模式: 'stretch'拉伸或'tile'平铺 |
| line_spacing | int | 20 | 行间距 |
| char_spacing | int | 0 | 字符间距 |
| fps | int | 30 | 视频帧率 |
| scroll_speed | float | 1 | 滚动速度(每秒滚动的行数) |
| audio_path | str | None | 背景音乐路径 |
| top_margin | int | 10 | 上边距(像素) |
| bottom_margin | int | 10 | 下边距(像素) |
| left_margin | int | 10 | 左边距(像素) |
| right_margin | int | 10 | 右边距(像素) |

## 使用方法

### 使用Crop滤镜方法

```python
from app.services.roll_video.roll_video_service import RollVideoService

service = RollVideoService()
result = service.create_roll_video_crop(
    text="要滚动的文本内容",
    output_path="output.mp4",
    width=720,
    height=1280,
    font_path="方正黑体简体.ttf",
    font_size=30,
    font_color=[0,0,0],
    bg_color=[255,255,255,1.0],  # 不透明白色背景
    line_spacing=20,
    char_spacing=10,
    fps=30,
    scroll_speed=1,
    top_margin=10,
    bottom_margin=10,
    left_margin=10,
    right_margin=10,
    background_url="https://example.com/bg.jpg",  # 背景图片URL
    scale_mode="stretch"  # 背景图缩放模式
)
```

### 使用Overlay CUDA方法

```python
from app.services.roll_video.roll_video_service import RollVideoService

service = RollVideoService()
result = service.create_roll_video_overlay_cuda(
    text="要滚动的文本内容",
    output_path="output.mp4",
    width=720,
    height=1280,
    font_path="方正黑体简体.ttf",
    font_size=30,
    font_color=[0,0,0],
    bg_color=[255,255,255,1.0],  # 不透明白色背景
    line_spacing=20,
    char_spacing=10,
    fps=30,
    scroll_speed=1,
    top_margin=10,
    bottom_margin=10,
    left_margin=10,
    right_margin=10,
    background_url="https://example.com/bg.jpg",  # 背景图片URL
    scale_mode="tile"  # 背景图缩放模式
)
```

## 注意事项

- 滚动速度越大，视频时长越短。
- **GPU 加速依赖**: 若需使用GPU加速（仅限不透明视频），请确保：
  - 安装了兼容的 Nvidia 显卡及最新驱动。
  - 安装了支持 NVENC 的 FFmpeg 版本。
- overlay_cuda 方法仍在优化中，某些情况下可能存在问题。
