# 滚动视频制作服务

## 需求：
   我想根据【参数】制作一个滚动视频。
整体方案：
   把文字排版成一张超长的图片，然后在固定尺寸的窗口（宽x高）里，通过控制这张图片向上移动，录制成视频。例如用Pillow生成文字画布，MoviePy做滚动。或者你有更好的方案也可以使用。先帮我实现CPU的效果。
    1、根据视频的宽度、字体、字体颜色、字号、字间距、行间距  计算每一行的文字与换行
    2、把每一行的文字渲染到指定宽度的图片上，图片要根据 视频背景颜色设置好
    3、根据滚动速度和视频高度，控制图片向上滚动
    4、把整个图片滚动完成后，渲染视频结束

## 实现方案

该服务使用Python实现，主要依赖以下库：
- PIL (Pillow): 用于文字渲染和图片处理
- MoviePy: 用于视频生成和处理
- NumPy: 用于数据处理

## 功能特点

- 支持自定义视频尺寸、字体、颜色等参数
- 自动处理文本换行和排版
- 平滑的滚动效果
- 支持添加背景音乐
- 支持从文件读取文本内容

## 使用方法

### 通过代码调用

```python
from app.services.roll_video.roll_video_service import RollVideoService

# 创建服务实例
service = RollVideoService()

# 创建滚动视频
result = service.create_roll_video(
    text="要展示的文本内容",
    output_path="output.mp4",
    width=1080,                              # 视频宽度
    height=1920,                             # 视频高度
    font_path="/path/to/font.ttf",           # 可选，字体路径
    font_size=40,                            # 字体大小
    font_color=(255, 255, 255),              # 字体颜色 (RGB)
    bg_color=(0, 0, 0),                      # 背景颜色 (RGB)
    line_spacing=20,                         # 行间距
    char_spacing=0,                          # 字符间距
    fps=30,                                  # 视频帧率
    scroll_speed=2,                          # 滚动速度(像素/帧)
    audio_path="/path/to/audio.mp3"          # 可选，背景音乐路径
)

# 输出结果
print(result)
```

### 通过命令行工具

```bash
# 使用文本字符串
python -m app.services.roll_video.cli --text "要展示的文本内容" --output output.mp4

# 使用文本文件
python -m app.services.roll_video.cli --text path/to/text.txt --output output.mp4

# 使用更多自定义参数
python -m app.services.roll_video.cli \
    --text "要展示的文本内容" \
    --output output.mp4 \
    --width 1080 \
    --height 1920 \
    --font-path /path/to/font.ttf \
    --font-size 40 \
    --font-color "255,255,255" \
    --bg-color "0,0,0" \
    --line-spacing 20 \
    --char-spacing 0 \
    --fps 30 \
    --scroll-speed 2 \
    --audio /path/to/audio.mp3
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | str | 必填 | 要展示的文本内容 |
| output_path | str | 必填 | 输出视频的路径 |
| width | int | 1080 | 视频宽度 |
| height | int | 1920 | 视频高度 |
| font_path | str | 系统默认 | 字体文件路径 |
| font_size | int | 40 | 字体大小 |
| font_color | (r,g,b) | (255,255,255) | 字体颜色 |
| bg_color | (r,g,b) | (0,0,0) | 背景颜色 |
| line_spacing | int | 20 | 行间距 |
| char_spacing | int | 0 | 字符间距 |
| fps | int | 30 | 视频帧率 |
| scroll_speed | int | 2 | 滚动速度(像素/帧) |
| audio_path | str | None | 背景音乐路径 |

## 依赖安装

```bash
pip install pillow moviepy numpy
```

## 注意事项

- 滚动速度越大，视频时长越短
- 文本内容较多时，可能需要较长的处理时间
- 如果使用中文字体，请确保指定了正确的字体文件路径

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




