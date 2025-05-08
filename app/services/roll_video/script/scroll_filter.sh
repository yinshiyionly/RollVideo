#!/bin/bash

# 使用scroll滤镜的滚动视频生成脚本
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
SLICE_HEIGHT=1000  # 每个切片的高度

# 创建临时目录
TEMP_DIR="./temp_scroll_files"
mkdir -p "$TEMP_DIR"

# 清理旧文件
rm -f $TEMP_DIR/*.png $TEMP_DIR/*.mp4

# 分析输入图像
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
  # 更新图像高度（宽度已调整为BACKGROUND_WIDTH）
  IMAGE_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$IMAGE_PATH")
fi

# 计算需要的切片数量
SLICE_COUNT=$(( (IMAGE_HEIGHT + SLICE_HEIGHT - 1) / SLICE_HEIGHT ))
echo "图像高度: $IMAGE_HEIGHT 像素，将分成 $SLICE_COUNT 个切片处理"

# 第1步：将输入图像分成多个切片以避免内存溢出
echo "将图像切割为 $SLICE_COUNT 个切片..."

for ((i=0; i<SLICE_COUNT; i++)); do
  # 计算当前切片的Y起始位置和高度
  Y_POS=$((i * SLICE_HEIGHT))
  CURRENT_HEIGHT=$SLICE_HEIGHT
  
  # 如果是最后一个切片，可能需要调整高度
  REMAINING_HEIGHT=$((IMAGE_HEIGHT - Y_POS))
  if [ $REMAINING_HEIGHT -lt $CURRENT_HEIGHT ]; then
    CURRENT_HEIGHT=$REMAINING_HEIGHT
  fi
  
  # 裁剪图像切片
  ffmpeg -y -loglevel error -i "$IMAGE_PATH" -vf "crop=$BACKGROUND_WIDTH:$CURRENT_HEIGHT:0:$Y_POS" "$TEMP_DIR/slice_$i.png"
  echo "  切片 $i: 从Y=$Y_POS 裁剪高度=$CURRENT_HEIGHT"
done

# 第2步：为每个切片创建滚动视频
echo "为每个切片创建滚动视频..."

# 计算滚动参数
TOTAL_HEIGHT=$(echo "$IMAGE_HEIGHT" | bc -l)
SCROLL_DURATION=$(echo "$TOTAL_DURATION - $START_SCROLL_TIME" | bc -l)
TOTAL_SCROLL_HEIGHT=$(echo "$TOTAL_HEIGHT + $BACKGROUND_HEIGHT" | bc -l)
SCROLL_SPEED=$(echo "$TOTAL_SCROLL_HEIGHT / $SCROLL_DURATION" | bc -l)
SCROLL_SPEED_PER_FRAME=$(echo "$SCROLL_SPEED / $FPS" | bc -l)

echo "滚动参数: 总高度=$TOTAL_HEIGHT, 滚动速度=$SCROLL_SPEED px/s ($SCROLL_SPEED_PER_FRAME px/frame)"

# 为每个切片创建滚动视频
for ((i=0; i<SLICE_COUNT; i++)); do
  SLICE_PATH="$TEMP_DIR/slice_$i.png"
  SLICE_VIDEO="$TEMP_DIR/slice_video_$i.mp4"
  
  # 检查切片是否存在
  if [ ! -f "$SLICE_PATH" ]; then
    echo "错误: 切片 $i 不存在，跳过处理"
    continue
  fi
  
  # 获取切片高度
  SLICE_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SLICE_PATH")
  CURRENT_HEIGHT=$SLICE_INFO
  echo "  处理切片 $i (高度=$CURRENT_HEIGHT)..."
  
  # 计算切片的滚动起始和结束位置
  Y_POS=$((i * SLICE_HEIGHT))
  START_VISIBLE_TIME=$(echo "scale=3; $START_SCROLL_TIME + ($Y_POS / $SCROLL_SPEED)" | bc -l)
  END_VISIBLE_TIME=$(echo "scale=3; $START_SCROLL_TIME + (($Y_POS + $CURRENT_HEIGHT) / $SCROLL_SPEED)" | bc -l)
  
  # 尝试使用scroll滤镜
  echo "  应用scroll滤镜 (可见时间 $START_VISIBLE_TIME - $END_VISIBLE_TIME)..."
  
  # 构建复杂滤镜链
  FILTER_COMPLEX="[0:v]fps=$FPS,scale=$BACKGROUND_WIDTH:$CURRENT_HEIGHT,setsar=1[img];"
  FILTER_COMPLEX+="color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$TOTAL_DURATION[bg];"
  FILTER_COMPLEX+="[img]setpts=PTS-STARTPTS+${START_SCROLL_TIME}/TB[img1];"
  
  # 尝试使用scroll滤镜
  if ffmpeg -y -loglevel error -i "$SLICE_PATH" \
    -filter_complex "$FILTER_COMPLEX[img1]split[a][b];[a]scroll=vertical=1:h=$BACKGROUND_HEIGHT:start_time=${START_SCROLL_TIME}:speed=${SCROLL_SPEED}[scr];[bg][scr]overlay=0:0[v]" \
    -map "[v]" -c:v h264_nvenc -preset p4 -t $TOTAL_DURATION \
    -b:v 8M -maxrate 10M -bufsize 10M \
    -pix_fmt yuv420p "$SLICE_VIDEO" 2>/dev/null; then
    echo "  滚动滤镜成功应用到切片 $i"
  else
    echo "  滚动滤镜失败，切换到传统覆盖方法..."
    
    # 如果scroll滤镜失败，使用传统的overlay方法
    # 修复表达式中的引号问题
    Y_OFFSET=$((i * SLICE_HEIGHT))
    
    # 创建一个没有特殊字符的表达式，避免引号问题
    ffmpeg -y -loglevel error \
      -f lavfi -i "color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$TOTAL_DURATION" \
      -loop 1 -i "$SLICE_PATH" \
      -filter_complex "[1:v]fps=$FPS,setpts=PTS-STARTPTS[img]; \
                       [0:v][img]overlay=0:if(lt(t,${START_SCROLL_TIME}),0,min(${BACKGROUND_HEIGHT}-h,-(t-${START_SCROLL_TIME})*${SCROLL_SPEED}+${BACKGROUND_HEIGHT}-${Y_OFFSET}))[v]" \
      -map "[v]" -c:v h264_nvenc -preset p4 -t $TOTAL_DURATION \
      -b:v 8M -maxrate 10M -bufsize 10M \
      -pix_fmt yuv420p "$SLICE_VIDEO"
    
    echo "  传统覆盖方法应用到切片 $i"
  fi
done

# 第3步：合并所有切片视频
echo "合并所有切片视频..."

# 创建白色背景视频
BACKGROUND_VIDEO="$TEMP_DIR/background.mp4"
ffmpeg -y -loglevel error \
  -f lavfi -i "color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$TOTAL_DURATION" \
  -c:v h264_nvenc -preset p1 -pix_fmt yuv420p "$BACKGROUND_VIDEO"

# 检查每个切片视频是否存在
ALL_SLICE_VIDEOS_EXIST=true
for ((i=0; i<SLICE_COUNT; i++)); do
  if [ ! -f "$TEMP_DIR/slice_video_$i.mp4" ]; then
    echo "警告: 切片视频 $i 不存在"
    ALL_SLICE_VIDEOS_EXIST=false
  fi
done

if [ "$ALL_SLICE_VIDEOS_EXIST" = true ]; then
  # 构建复杂滤镜链合并所有切片
  FILTER_COMPLEX=""
  OVERLAY_CHAIN="[0:v]"
  
  for ((i=0; i<SLICE_COUNT; i++)); do
    SLICE_VIDEO="$TEMP_DIR/slice_video_$i.mp4"
    SLICE_INDEX=$((i + 1))
    FILTER_COMPLEX+="[$SLICE_INDEX:v]format=yuva420p,setpts=PTS-STARTPTS[v$i];"
    OVERLAY_CHAIN+="[v$i]overlay=0:0:shortest=0"
    
    if [ $i -lt $((SLICE_COUNT - 1)) ]; then
      OVERLAY_CHAIN+="[o$i];"
      NEXT_INPUT="[o$i]"
    else
      OVERLAY_CHAIN+="[out]"
    fi
  done
  
  echo "合并 $SLICE_COUNT 个切片视频..."
  
  # 构建输入文件列表
  INPUT_FILES="-i $BACKGROUND_VIDEO"
  for ((i=0; i<SLICE_COUNT; i++)); do
    INPUT_FILES+=" -i $TEMP_DIR/slice_video_$i.mp4"
  done
  
  # 最终合并命令
  ffmpeg -y -loglevel error $INPUT_FILES \
    -filter_complex "$FILTER_COMPLEX$OVERLAY_CHAIN" \
    -map "[out]" -c:v h264_nvenc -preset p4 \
    -b:v 10M -maxrate 12M -bufsize 12M \
    -pix_fmt yuv420p "$OUTPUT_VIDEO"
  
  if [ $? -eq 0 ]; then
    echo "滚动视频生成成功！"
    
    # 清理临时文件
    rm -rf "$TEMP_DIR"
    
    exit 0
  else
    echo "合并视频失败"
    exit 1
  fi
else
  echo "错误: 有切片视频缺失，无法完成合并"
  exit 1
fi 