# 使用FFmpeg与overlay_cuda加速实现长文本滚动视频

## 实现步骤

#### 1. 文本图片准备

- 可以直接使用 renderer目录下的text_renderer.py 直接渲染出图片
  
#### 2. GPU加速滚动实现

基本命令结构：

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
  -f lavfi -i "color=c=[背景色]:s=[宽]x[高]:r=[帧率],format=yuv420p,hwupload_cuda" \
  -i [长文字图片路径] \
  -filter_complex "\
    [1:v]format=rgba,hwupload_cuda[img]; \
    [0:v][img]overlaycuda=x=0:y='[滚动公式]':shortest=1[out]" \
  -map "[out]" -c:v h264_nvenc -preset [质量预设] -b:v [码率] \
  -t [时长] [输出文件名]
```

#### 3. 滚动算法实现

##### 基础匀速滚动

```
y='min(0, -(h-[视频高度])+t*[速度])'
```

- `h` - 原图高度
- `[视频高度]` - 输出视频的高度（如1280）
- `[速度]` - 滚动速度（像素/秒，建议40-60）

##### 带加速减速效果的滚动

```
y='if(lt(t,[加速时间]),-[加速系数]*t*t,-[总高度]+[视频高度]+if(gt(t,[减速开始时间]),-(T-t)*(T-t)*[减速系数],t*[中间速度]))'
```

其中：
- `[加速时间]` - 开始的加速阶段时长（秒）
- `[加速系数]` - 加速曲线系数
- `[总高度]` - 原图高度
- `[减速开始时间]` - 开始减速的时间点
- `T` - 视频总时长
- `[减速系数]` - 减速曲线系数
- `[中间速度]` - 匀速阶段的速度（像素/秒）

####45. 视频质量优化

##### 编码器参数优化

```bash
-c:v h264_nvenc -preset p7 -tune hq -b:v 8M -bufsize 12M -rc vbr_hq
```

或使用固定量化参数提高文字清晰度：

```bash
-c:v h264_nvenc -preset p7 -rc constqp -qp 18
```

##### 帧率与分辨率

- 帧率：30fps适合大多数情况，60fps提供更平滑滚动
- 分辨率：保持与排版图片宽度一致（如720px），高度根据设备确定（如1280px）

### 完整实现代码示例

#### 基础版本（简单滚动）

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
  -f lavfi -i "color=c=#1a1a1a:s=720x1280:r=30,format=yuv420p,hwupload_cuda" \
  -i 长文字图片.png \
  -filter_complex "\
    [1:v]format=rgba,hwupload_cuda[img]; \
    [0:v][img]overlaycuda=x=0:y='min(0,-(h-1280)+t*60)':shortest=1[out]" \
  -map "[out]" -c:v h264_nvenc -preset p7 -tune hq -b:v 5M \
  -t 240 文字滚动视频.mp4
```

#### 高级版本（带加速减速效果）

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
  -f lavfi -i "color=c=#1a1a1a:s=720x1280:r=60,format=yuv420p,hwupload_cuda" \
  -i 长文字图片.png \
  -filter_complex "\
    [1:v]format=rgba,hwupload_cuda[img]; \
    [0:v][img]overlaycuda=x=0:y='if(lt(t,3),-2*t*t,-12800+1280+if(gt(t,235),-(240-t)*(240-t)*5,t*50))':eof_action=endall:shortest=1[out]" \
  -map "[out]" -c:v h264_nvenc -preset p7 -tune hq -b:v 8M -bufsize 12M -rc vbr_hq \
  -t 240 文字滚动视频.mp4
```

#### 带背景音乐版本

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
  -f lavfi -i "color=c=#1a1a1a:s=720x1280:r=60,format=yuv420p,hwupload_cuda" \
  -i 长文字图片.png \
  -i 背景音乐.mp3 \
  -filter_complex "\
    [1:v]format=rgba,hwupload_cuda[img]; \
    [0:v][img]overlaycuda=x=0:y='if(lt(t,3),-2*t*t,-12800+1280+t*50)':eof_action=endall:shortest=1[out]" \
  -map "[out]" -map 2:a -c:v h264_nvenc -preset p7 -tune hq -b:v 8M \
  -c:a aac -b:a 192k -t 240 文字滚动视频.mp4
```

### 扩展功能实现

#### 字幕叠加

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda \
  -f lavfi -i "color=c=#1a1a1a:s=720x1280:r=30,format=yuv420p,hwupload_cuda" \
  -i 长文字图片.png \
  -filter_complex "\
    [1:v]format=rgba,hwupload_cuda[img]; \
    [0:v][img]overlaycuda=x=0:y='min(0,-(h-1280)+t*50)':shortest=1, \
    drawtext=fontfile=/path/to/font.ttf:text='标题':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=50[out]" \
  -map "[out]" -c:v h264_nvenc -preset p7 -tune hq -b:v 5M \
  -t 240 文字滚动视频.mp4
```

#### 淡入淡出效果

可在开头或结尾添加淡入淡出效果：

```bash
-filter_complex "\
  [1:v]format=rgba,hwupload_cuda[img]; \
  [0:v][img]overlaycuda=x=0:y='min(0,-(h-1280)+t*50)':shortest=1, \
  fade=in:0:30,fade=out:210:30[out]"
```

## 命令行参数说明

### 必要参数

- `-hwaccel cuda -hwaccel_output_format cuda` - 启用CUDA硬件加速
- `-f lavfi -i "color=c=#颜色:s=宽x高:r=帧率"` - 创建背景
- `-i 图片路径` - 指定长文字图片路径
- `format=rgba,hwupload_cuda` - 将图片上传到GPU并保留透明通道
- `overlaycuda=x=0:y='滚动公式'` - GPU加速的叠加滤镜及滚动控制
- `-c:v h264_nvenc` - 使用NVIDIA硬件编码器

### 可选参数及建议值

`-preset p4`
"-progress", "pipe:2",  # 输出进度信息到stderr
"-stats",  # 启用统计信息
"-stats_period", "1",  # 每1秒输出一次统计信息

其他参数可参考 renderer.video_renderder.py 中的 ffmpeg cmd 相关的定义


## 代码实现建议

建议开发一个Python脚本，用于处理参数并调用FFmpeg命令。脚本应包含：

1. 参数解析（图片路径、输出路径、滚动速度等）
2. 自动计算最佳视频长度
3. 检测GPU和CUDA可用性
4. 执行优化的FFmpeg命令
5. 进度监控和错误处理

## 注意

1、要充分使用GPU加速的效果提升渲染效率。
2、当GPU不可用或滤镜不可用时要有CPU渲染的回退策略
3、可以使用的硬件资源：8核、30G内存、10G显存(A10卡)