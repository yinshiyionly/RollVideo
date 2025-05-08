#!/bin/bash

# 简化的滚动视频生成脚本 - 使用简单scroll滤镜
# 输入参数
IMAGE_PATH="$1"
OUTPUT_VIDEO="$2"
VERTICAL_SCROLL=${3:-0.01}  # 默认滚动速度0.01，可通过第三个参数修改

# 基础设置
FPS=30
DURATION=300  # 视频时长，固定300秒
START_TIME=$(date +%s.%N)

# 分析图像高度
echo "分析输入图像..."
IMAGE_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=,:p=0 "$IMAGE_PATH")
IMAGE_WIDTH=$(echo $IMAGE_INFO | cut -d, -f1)
IMAGE_HEIGHT=$(echo $IMAGE_INFO | cut -d, -f2)
echo "原始图像尺寸: ${IMAGE_WIDTH}x${IMAGE_HEIGHT} 像素"

# 检查尺寸是否符合要求（宽高都必须是2的倍数）
NEED_ADJUST=false
NEW_WIDTH=$IMAGE_WIDTH
NEW_HEIGHT=$IMAGE_HEIGHT

# 确保宽高都是偶数（2的倍数）
if [ $((IMAGE_WIDTH % 2)) -ne 0 ]; then
  NEW_WIDTH=$((IMAGE_WIDTH + 1))
  NEED_ADJUST=true
fi
if [ $((IMAGE_HEIGHT % 2)) -ne 0 ]; then
  NEW_HEIGHT=$((IMAGE_HEIGHT + 1))
  NEED_ADJUST=true
fi

# 检查图像高度是否超过NVENC最大支持（4096像素）
MAX_HEIGHT=4000  # 略小于4096以确保安全
USE_CPU_ENCODER=false

if [ $NEW_HEIGHT -gt $MAX_HEIGHT ]; then
  echo "警告: 图像高度 $NEW_HEIGHT 超过NVENC最大支持高度 $MAX_HEIGHT"
  echo "将使用CPU编码器(libx264)以保留所有图像内容"
  USE_CPU_ENCODER=true
fi

echo "使用滚动速度: $VERTICAL_SCROLL (每帧屏幕比例)"

# 构建滤镜字符串
FILTER="fps=$FPS"

# 如果需要调整尺寸以满足编码器要求
if [ "$NEED_ADJUST" = true ]; then
  echo "调整图像尺寸以满足编码器要求 (必须是2的倍数): ${NEW_WIDTH}x${NEW_HEIGHT}"
  # 使用pad而不是scale，以便保留所有内容
  FILTER="$FILTER,pad=$NEW_WIDTH:$NEW_HEIGHT:0:0:black"
fi

# 添加其他滤镜
FILTER="$FILTER,scroll=vertical=$VERTICAL_SCROLL:h=1,format=yuv420p"

echo "使用滤镜: $FILTER"

# 根据图像高度选择合适的编码器
if [ "$USE_CPU_ENCODER" = true ]; then
  echo "使用CPU编码器处理超高图像..."
  
  # 使用libx264编码器
  ffmpeg -y -i "$IMAGE_PATH" \
    -vf "$FILTER" \
    -c:v libx264 -preset medium -crf 23 -t $DURATION \
    -maxrate 8M -bufsize 10M \
    "$OUTPUT_VIDEO"
else
  echo "使用NVENC硬件加速编码器..."
  
  # 使用NVENC硬件加速
  ffmpeg -y -i "$IMAGE_PATH" \
    -vf "$FILTER" \
    -c:v h264_nvenc -preset p4 -t $DURATION \
    -b:v 8M -maxrate 10M -bufsize 10M \
    "$OUTPUT_VIDEO"
fi

# 保存ffmpeg的返回值
FFMPEG_RESULT=$?

# 计算处理时间
END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

# 输出结果
if [ $FFMPEG_RESULT -eq 0 ]; then
  echo "滚动视频处理成功！"
  echo "处理时间: ${ELAPSED_TIME}秒"
  echo "输出视频: $OUTPUT_VIDEO"
  exit 0
else
  echo "滚动视频处理失败（错误码: $FFMPEG_RESULT）"
  echo "如果您仍然遇到问题，请尝试使用更小的滚动速度(如0.001)或降低输入图像分辨率。"
  exit 1
fi 