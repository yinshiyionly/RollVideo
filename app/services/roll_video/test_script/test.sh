#!/bin/bash

# 输入参数
IMAGE_PATH="$1"
OUTPUT_VIDEO="$2"

# 基础设置
FPS=30
BACKGROUND_WIDTH=720
BACKGROUND_HEIGHT=1280
TOTAL_DURATION=300
START_SCROLL_TIME=5
THREADS=4
SLICE_HEIGHT=1000

# 临时文件夹
TEMP_DIR="./temp_scroll_files"
mkdir -p "$TEMP_DIR"

# 清理旧文件
rm -f $TEMP_DIR/slice_*.png $TEMP_DIR/part_*.mp4

# 切割图像为多个部分
echo "切割图像为多个部分..."
for i in {0..3}; do
  START_Y=$((i * SLICE_HEIGHT))
  ffmpeg -y -i "$IMAGE_PATH" -vf "crop=720:${SLICE_HEIGHT}:0:${START_Y}" "$TEMP_DIR/slice_${i}.png"
done

# 计算滚动参数
TOTAL_HEIGHT=$((SLICE_HEIGHT * 4))
SCROLL_DURATION=$(echo "$TOTAL_DURATION - $START_SCROLL_TIME" | bc -l)
TOTAL_SCROLL_HEIGHT=$(echo "$TOTAL_HEIGHT + $BACKGROUND_HEIGHT" | bc -l)
SCROLL_SPEED=$(echo "$TOTAL_SCROLL_HEIGHT / $SCROLL_DURATION" | bc -l)

echo "处理参数：高度=$TOTAL_HEIGHT, 滚动速度=$SCROLL_SPEED"

# 使用最简单而稳定的方法：CPU处理叠加，GPU编码
echo "使用最简单的CPU+GPU混合方法进行处理..."

# 为每个切片创建带滚动效果的独立视频
for i in {0..3}; do
  OFFSET=$((i * SLICE_HEIGHT))
  echo "处理切片 $i..."
  
  # 使用CPU处理，但用GPU编码
  ffmpeg -y \
  -loop 1 -framerate $FPS -i "$TEMP_DIR/slice_${i}.png" \
  -f lavfi -i "color=c=black:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS" \
  -filter_complex "\
  [1:v][0:v]overlay=x=(main_w-overlay_w)/2:y='if(lt(t,$START_SCROLL_TIME),$OFFSET,($OFFSET-(t-$START_SCROLL_TIME)*$SCROLL_SPEED))'[v]" \
  -map "[v]" \
  -c:v h264_nvenc -preset p5 -pix_fmt yuv420p -t $TOTAL_DURATION \
  -b:v 10M -maxrate 10M -bufsize 10M \
  "$TEMP_DIR/part_${i}.mp4"
  
  if [ $? -ne 0 ]; then
    echo "处理切片 $i 失败"
    rm -rf "$TEMP_DIR"
    exit 1
  fi
done

# 合并所有切片
echo "合并所有切片..."

# 创建背景
ffmpeg -y -f lavfi -i "color=c=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$TOTAL_DURATION" "$TEMP_DIR/background.mp4"

# 检查所有切片文件是否存在
for i in {0..3}; do
  if [ ! -f "$TEMP_DIR/part_${i}.mp4" ]; then
    echo "切片 $i 的文件丢失，无法合并"
    rm -rf "$TEMP_DIR"
    exit 1
  fi
done

# 合并切片
ffmpeg -y \
-i "$TEMP_DIR/background.mp4" \
-i "$TEMP_DIR/part_0.mp4" \
-i "$TEMP_DIR/part_1.mp4" \
-i "$TEMP_DIR/part_2.mp4" \
-i "$TEMP_DIR/part_3.mp4" \
-filter_complex "\
[0:v][1:v]overlay=shortest=1[bg0]; \
[bg0][2:v]overlay=shortest=1[bg1]; \
[bg1][3:v]overlay=shortest=1[bg2]; \
[bg2][4:v]overlay=shortest=1[v]" \
-map "[v]" \
-c:v h264_nvenc -preset p5 -pix_fmt yuv420p \
-b:v 10M -maxrate 10M -bufsize 10M \
"$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
  echo "处理成功完成！"
  rm -rf "$TEMP_DIR"
  exit 0
else
  echo "合并切片失败"
  rm -rf "$TEMP_DIR"
  exit 1
fi