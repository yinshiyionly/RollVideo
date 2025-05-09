#!/bin/bash

# 基于CUDA加速但不预切割图像的滚动视频生成脚本
# 适用于高内存高显存系统
# 输入参数
IMAGE_PATH="$1"
OUTPUT_VIDEO="$2"

# 基础设置
FPS=30
BACKGROUND_WIDTH=720
BACKGROUND_HEIGHT=1280
TOTAL_DURATION=300
START_SCROLL_TIME=5
MAX_PARALLEL_FRAMES=12  # 并行处理的最大帧数

# 临时文件夹
TEMP_DIR="./temp_no_slice_cuda"
mkdir -p "$TEMP_DIR"
mkdir -p "$TEMP_DIR/cuda_frames"

# 清理旧文件
rm -f $TEMP_DIR/cuda_frames/* $TEMP_DIR/*.mp4 $TEMP_DIR/*.txt

# 计算滚动参数 - 总高度将从图像提取
echo "分析输入图像..."
IMAGE_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=,:p=0 "$IMAGE_PATH")
IMAGE_WIDTH=$(echo $IMAGE_INFO | cut -d, -f1)
IMAGE_HEIGHT=$(echo $IMAGE_INFO | cut -d, -f2)

# 确保宽度与设置匹配
if [ "$IMAGE_WIDTH" -ne "$BACKGROUND_WIDTH" ]; then
  echo "调整图像宽度从 $IMAGE_WIDTH 到 $BACKGROUND_WIDTH..."
  TEMP_IMAGE="$TEMP_DIR/resized_input.png"
  ffmpeg -y -i "$IMAGE_PATH" -vf "scale=$BACKGROUND_WIDTH:$IMAGE_HEIGHT" "$TEMP_IMAGE"
  IMAGE_PATH="$TEMP_IMAGE"
  IMAGE_WIDTH=$BACKGROUND_WIDTH
fi

# 计算滚动参数
SCROLL_DURATION=$(echo "$TOTAL_DURATION - $START_SCROLL_TIME" | bc -l)
TOTAL_SCROLL_HEIGHT=$(echo "$IMAGE_HEIGHT + $BACKGROUND_HEIGHT" | bc -l)
SCROLL_SPEED=$(echo "$TOTAL_SCROLL_HEIGHT / $SCROLL_DURATION" | bc -l)
TOTAL_FRAMES=$((TOTAL_DURATION * FPS))

echo "处理参数：图像宽度=$IMAGE_WIDTH, 图像高度=$IMAGE_HEIGHT, 滚动速度=$SCROLL_SPEED, 总帧数=$TOTAL_FRAMES"

# 直接使用CUDA处理每一帧
echo "使用CUDA处理帧 (不预切割)..."

process_frame_cuda() {
  local FRAME=$1
  local TIME=$(echo "scale=3; $FRAME / $FPS" | bc -l)
  
  # 计算当前帧应该显示的Y位置
  if (( $(echo "$TIME < $START_SCROLL_TIME" | bc -l) )); then
    # 滚动前固定在顶部
    Y_POS=0
  else
    # 滚动中根据时间计算位置
    ELAPSED=$(echo "$TIME - $START_SCROLL_TIME" | bc -l)
    Y_POS=$(echo "$ELAPSED * $SCROLL_SPEED" | bc -l)
    # 确保Y位置不超过图像高度
    if (( $(echo "$Y_POS > ($IMAGE_HEIGHT - $BACKGROUND_HEIGHT)" | bc -l) )); then
      Y_POS=$(echo "$IMAGE_HEIGHT - $BACKGROUND_HEIGHT" | bc -l)
    fi
  fi
  
  # 将位置转为整数
  Y_POS_INT=$(printf "%.0f" $Y_POS)
  
  # 直接使用CUDA加速裁剪并处理帧
  # 这里我们在一个FFmpeg命令中完成裁剪和合成，避免中间文件
  ffmpeg -y -loglevel error \
    -hwaccel cuda -hwaccel_output_format cuda \
    -i "$IMAGE_PATH" \
    -f lavfi -i "color=c=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=1" \
    -filter_complex "[0:v]format=nv12,hwupload_cuda,crop=${BACKGROUND_WIDTH}:${BACKGROUND_HEIGHT}:0:${Y_POS_INT}[fg]; \
                     [1:v]format=nv12,hwupload_cuda[bg]; \
                     [bg][fg]overlay_cuda=0:0[out]" \
    -map "[out]" -frames:v 1 \
    -c:v h264_nvenc -preset p1 -pix_fmt yuv420p \
    "$TEMP_DIR/cuda_frames/frame_$FRAME.jpg"
    
  echo -ne "已处理: $FRAME/$TOTAL_FRAMES 帧 ($(printf "%.1f" $(echo "scale=3; $FRAME * 100 / $TOTAL_FRAMES" | bc -l))%)\r"
}

export -f process_frame_cuda
export IMAGE_PATH TEMP_DIR BACKGROUND_WIDTH BACKGROUND_HEIGHT START_SCROLL_TIME SCROLL_SPEED IMAGE_HEIGHT FPS

# 并行处理CUDA帧
echo "开始并行处理帧..."
for ((frame=0; frame<TOTAL_FRAMES; frame+=$MAX_PARALLEL_FRAMES)); do
  BATCH_END=$((frame + MAX_PARALLEL_FRAMES - 1))
  if [ $BATCH_END -ge $TOTAL_FRAMES ]; then
    BATCH_END=$((TOTAL_FRAMES - 1))
  fi
  
  # 为当前批次创建并行任务
  for ((f=frame; f<=BATCH_END; f++)); do
    (process_frame_cuda $f) &
  done
  
  # 等待当前批次完成
  wait
done

echo -e "\n所有帧处理完成"

# 检查生成的帧数量
FRAMES_CREATED=$(ls -1 "$TEMP_DIR/cuda_frames/" | wc -l)
if [ $FRAMES_CREATED -ne $TOTAL_FRAMES ]; then
  echo "警告: 生成的帧数量 ($FRAMES_CREATED) 与预期 ($TOTAL_FRAMES) 不符"
fi

# 创建最终视频
echo "生成最终视频..."

# 创建帧列表文件
FRAMES_LIST="$TEMP_DIR/frames_list.txt"
> $FRAMES_LIST

for ((frame=0; frame<TOTAL_FRAMES; frame++)); do
  if [ -f "$TEMP_DIR/cuda_frames/frame_$frame.jpg" ]; then
    echo "file '$TEMP_DIR/cuda_frames/frame_$frame.jpg'" >> $FRAMES_LIST
  else
    echo "警告: 帧 $frame 缺失"
    # 尝试使用最近的可用帧
    NEAREST_FRAME=$((frame - 1))
    while [ $NEAREST_FRAME -ge 0 ]; do
      if [ -f "$TEMP_DIR/cuda_frames/frame_$NEAREST_FRAME.jpg" ]; then
        echo "file '$TEMP_DIR/cuda_frames/frame_$NEAREST_FRAME.jpg'" >> $FRAMES_LIST
        echo "  使用帧 $NEAREST_FRAME 代替缺失帧 $frame"
        break
      fi
      NEAREST_FRAME=$((NEAREST_FRAME - 1))
    done
  fi
done

# 使用帧列表生成最终视频
ffmpeg -y -f concat -safe 0 -i "$FRAMES_LIST" \
  -c:v h264_nvenc -preset p5 -pix_fmt yuv420p -r $FPS \
  -b:v 10M -maxrate 10M -bufsize 10M \
  "$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
  echo "CUDA帧处理成功完成！"
  echo "总共处理了 $FRAMES_CREATED 帧"
  echo "输出视频：$OUTPUT_VIDEO"
  
  # 清理临时文件
  rm -rf "$TEMP_DIR"
  
  exit 0
else
  echo "视频生成失败"
  exit 1
fi 