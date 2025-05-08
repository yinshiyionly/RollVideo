#!/bin/bash

# 基于CUDA加速的长图滚动视频生成脚本 - 优化版
# 输入参数
IMAGE_PATH="$1"                  # 第一个参数：输入图像
OUTPUT_VIDEO="$2"                # 第二个参数：输出视频
SCROLL_SPEED_FACTOR=${3:-1}      # 第三个参数：滚动速度因子(1=24px/s, 2=48px/s)
METHOD=${4:-1}                   # 第四个参数：滚动方法选择(1=scroll, 2=zoompan, 3=overlay)

# 基础设置
FPS=24
BACKGROUND_WIDTH=720
BACKGROUND_HEIGHT=1280
START_SCROLL_TIME=5
TEMP_DIR="./temp_cuda_optimized"
mkdir -p "$TEMP_DIR"

# 开始计时（整体）
SCRIPT_START_TIME=$(date +%s.%N)

# 分析输入图像
echo "分析输入图像..."
IMAGE_ANALYSIS_START=$(date +%s.%N)

IMAGE_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=,:p=0 "$IMAGE_PATH")
IMAGE_WIDTH=$(echo $IMAGE_INFO | cut -d, -f1)
IMAGE_HEIGHT=$(echo $IMAGE_INFO | cut -d, -f2)

# 确保宽度与设置匹配
if [ "$IMAGE_WIDTH" -ne "$BACKGROUND_WIDTH" ]; then
  echo "调整图像宽度从 $IMAGE_WIDTH 到 $BACKGROUND_WIDTH..."
  TEMP_IMAGE="$TEMP_DIR/resized_input.png"
  ffmpeg -y -hwaccel cuda -i "$IMAGE_PATH" -vf "scale=$BACKGROUND_WIDTH:$IMAGE_HEIGHT" "$TEMP_IMAGE"
  if [ $? -eq 0 ]; then
    IMAGE_PATH="$TEMP_IMAGE"
  else
    echo "警告：CUDA缩放失败，使用CPU..."
  ffmpeg -y -i "$IMAGE_PATH" -vf "scale=$BACKGROUND_WIDTH:$IMAGE_HEIGHT" "$TEMP_IMAGE"
  IMAGE_PATH="$TEMP_IMAGE"
  fi
fi

# 计算以像素/秒为单位的速度
BASE_SCROLL_SPEED=24  # 每秒基础滚动像素数
SCROLL_SPEED=$(echo "$BASE_SCROLL_SPEED * $SCROLL_SPEED_FACTOR" | bc -l)

# 计算滚动需要的总时间
SCROLL_DURATION=$(echo "($IMAGE_HEIGHT + $BACKGROUND_HEIGHT) / $SCROLL_SPEED" | bc -l)
BUFFER_TIME=10  # 额外缓冲时间
TOTAL_DURATION=$(echo "$START_SCROLL_TIME + $SCROLL_DURATION + $BUFFER_TIME" | bc -l | xargs printf "%.0f")
echo "计算得到总视频时长: $TOTAL_DURATION 秒（包含前期静止 $START_SCROLL_TIME 秒和缓冲 $BUFFER_TIME 秒）"

IMAGE_ANALYSIS_END=$(date +%s.%N)
IMAGE_ANALYSIS_TIME=$(echo "$IMAGE_ANALYSIS_END - $IMAGE_ANALYSIS_START" | bc -l)
echo "图像分析耗时: ${IMAGE_ANALYSIS_TIME} 秒"
echo "处理参数：图像尺寸=${BACKGROUND_WIDTH}x${IMAGE_HEIGHT}, 滚动速度=${SCROLL_SPEED}px/s, 滚动时长=${SCROLL_DURATION}s"

# 统一帧切割处理 - 为所有方法准备图像块
echo "执行帧切割预处理..."
FRAME_CUTTING_START=$(date +%s.%N)

# 计算需要的分块数（每1000像素高度一个分块）
BLOCK_HEIGHT=1000
NUM_BLOCKS=$(( (IMAGE_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT ))
echo "将图像分为 $NUM_BLOCKS 个块处理"

# 为每个分块创建临时文件
mkdir -p "$TEMP_DIR/blocks"

# 切割图像为多个块
for ((i=0; i<NUM_BLOCKS; i++)); do
  Y_POS=$((i * BLOCK_HEIGHT))
  REMAIN_HEIGHT=$((IMAGE_HEIGHT - Y_POS))
  CURRENT_HEIGHT=$((REMAIN_HEIGHT < BLOCK_HEIGHT ? REMAIN_HEIGHT : BLOCK_HEIGHT))
  
  ffmpeg -y -loglevel error -i "$IMAGE_PATH" \
    -vf "crop=$BACKGROUND_WIDTH:$CURRENT_HEIGHT:0:$Y_POS" \
    "$TEMP_DIR/blocks/block_$i.png"
done

FRAME_CUTTING_END=$(date +%s.%N)
FRAME_CUTTING_TIME=$(echo "$FRAME_CUTTING_END - $FRAME_CUTTING_START" | bc -l)
echo "帧切割耗时: ${FRAME_CUTTING_TIME} 秒"

# 使用指定方法处理滚动
PROCESS_START=$(date +%s.%N)

# 输出方法名称
METHODS=("" "scroll滤镜" "zoompan滤镜" "overlay滤镜")
METHOD_NAME=${METHODS[$METHOD]}

echo "使用方法${METHOD}：${METHOD_NAME}处理"

case $METHOD in
  1)
    # 方法1：使用scroll滤镜 
    echo "使用方法1：scroll滤镜处理"
    
    # 创建滚动序列
    rm -f "$TEMP_DIR/concat_list.txt"
    
    # 第一个块 - 静止几秒后开始滚动
    BLOCK_DURATION=$(echo "$BLOCK_HEIGHT / $SCROLL_SPEED + $START_SCROLL_TIME" | bc -l)
    BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
    
    # 计算滚动速度（像素/秒转换为scroll滤镜参数）
    # scroll滤镜速度为v参数，每帧滚动的像素数
    PIXELS_PER_FRAME=$(echo "scale=6; $SCROLL_SPEED / $FPS" | bc -l)
    
    # 计算裁剪高度（不超过块高度）
    CROP_HEIGHT=$(( BACKGROUND_HEIGHT < BLOCK_HEIGHT ? BACKGROUND_HEIGHT : BLOCK_HEIGHT ))
    
    ffmpeg -y -loop 1 -i "$TEMP_DIR/blocks/block_0.png" \
      -f lavfi -i "color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}" \
      -filter_complex "
        [0:v]fps=$FPS,setpts=PTS-STARTPTS+$START_SCROLL_TIME/TB[delayed];
        [delayed]split[a][b];
        [a]trim=start=0:end=$START_SCROLL_TIME,setpts=PTS-STARTPTS,crop=$BACKGROUND_WIDTH:$CROP_HEIGHT:0:0[still];
        [b]trim=start=$START_SCROLL_TIME,setpts=PTS-STARTPTS,scroll=vertical=1:h=1:v=$PIXELS_PER_FRAME[scrolled];
        [still][scrolled]concat=n=2:v=1:a=0[out]
      " \
      -map "[out]" -t $BLOCK_DURATION_INT \
      -c:v h264_nvenc -preset p4 -r $FPS \
      "$TEMP_DIR/block_0_scroll.mp4"
    
    echo "file 'block_0_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    
    # 处理其余块
    for ((i=1; i<NUM_BLOCKS; i++)); do
      CURRENT_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$TEMP_DIR/blocks/block_$i.png")
      BLOCK_DURATION=$(echo "$CURRENT_HEIGHT / $SCROLL_SPEED" | bc -l)
      BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
      
      if [ $BLOCK_DURATION_INT -lt 1 ]; then
        BLOCK_DURATION_INT=1
      fi
      
      # 计算滚动速度
      PIXELS_PER_FRAME=$(echo "scale=6; $SCROLL_SPEED / $FPS" | bc -l)
      
      ffmpeg -y -loop 1 -i "$TEMP_DIR/blocks/block_$i.png" \
        -filter_complex "
          [0:v]fps=$FPS,scroll=vertical=1:h=1:v=$PIXELS_PER_FRAME
        " \
        -c:v h264_nvenc -preset p4 -r $FPS -t $BLOCK_DURATION_INT \
        "$TEMP_DIR/block_${i}_scroll.mp4"
      
      echo "file 'block_${i}_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    done
    ;;
    
  2)
    # 方法2：使用zoompan滤镜
    echo "使用方法2：zoompan滤镜处理"
    
    # 创建滚动序列
    rm -f "$TEMP_DIR/concat_list.txt"
    
    # 第一个块 - 静止几秒后开始滚动
    BLOCK_DURATION=$(echo "$BLOCK_HEIGHT / $SCROLL_SPEED + $START_SCROLL_TIME" | bc -l)
    BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
    
    # 计算每帧移动像素数
    PIXELS_PER_FRAME=$(echo "scale=6; $SCROLL_SPEED / $FPS" | bc -l)
    
    # 计算开始滚动的帧数
    START_FRAME=$((START_SCROLL_TIME * FPS))
    
    # 使用超级简化的zoompan表达式，用if替代min函数
    ffmpeg -y -loop 1 -i "$TEMP_DIR/blocks/block_0.png" \
      -vf "fps=$FPS,crop=$BACKGROUND_WIDTH:$BLOCK_HEIGHT:0:0,zoompan=d=1:x=0:y=if(lt(n\,$START_FRAME)\,0\,if(gt((n-$START_FRAME)*$PIXELS_PER_FRAME\,($BLOCK_HEIGHT-$BACKGROUND_HEIGHT))\,($BLOCK_HEIGHT-$BACKGROUND_HEIGHT)\,(n-$START_FRAME)*$PIXELS_PER_FRAME)):s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:fps=$FPS" \
      -c:v h264_nvenc -preset p4 -r $FPS -t $BLOCK_DURATION_INT \
      "$TEMP_DIR/block_0_scroll.mp4"
    
    echo "file 'block_0_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    
    # 处理其余块
    for ((i=1; i<NUM_BLOCKS; i++)); do
      CURRENT_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$TEMP_DIR/blocks/block_$i.png")
      BLOCK_DURATION=$(echo "$CURRENT_HEIGHT / $SCROLL_SPEED" | bc -l)
      BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
      
      if [ $BLOCK_DURATION_INT -lt 1 ]; then
        BLOCK_DURATION_INT=1
      fi
      
      # 后续块也用if替代min函数
      ffmpeg -y -loop 1 -i "$TEMP_DIR/blocks/block_$i.png" \
        -vf "fps=$FPS,crop=$BACKGROUND_WIDTH:$CURRENT_HEIGHT:0:0,zoompan=d=1:x=0:y=if(gt(n*$PIXELS_PER_FRAME\,($CURRENT_HEIGHT-$BACKGROUND_HEIGHT))\,($CURRENT_HEIGHT-$BACKGROUND_HEIGHT)\,n*$PIXELS_PER_FRAME):s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:fps=$FPS" \
        -c:v h264_nvenc -preset p4 -r $FPS -t $BLOCK_DURATION_INT \
        "$TEMP_DIR/block_${i}_scroll.mp4"
      
      echo "file '$TEMP_DIR/block_${i}_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    done
    ;;
    
  3)
    # 方法3：使用overlay滤镜处理 - 高性能分块处理版
    echo "使用方法3：overlay滤镜分块处理"
    
    # 创建滚动序列
    rm -f "$TEMP_DIR/concat_list.txt"
    
    # 第一个块 - 静止几秒后开始滚动
    BLOCK_DURATION=$(echo "$BLOCK_HEIGHT / $SCROLL_SPEED + $START_SCROLL_TIME" | bc -l)
    BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
    
    # 确保平滑滚动的表达式 - 计算精确移动位置
    MAX_SCROLL1=$(( BLOCK_HEIGHT - BACKGROUND_HEIGHT > 0 ? BLOCK_HEIGHT - BACKGROUND_HEIGHT : 0 ))
    EXPR1="if(lt(t\,$START_SCROLL_TIME)\,0\,(t-$START_SCROLL_TIME)*$SCROLL_SPEED)"
    
    echo "处理第一个块，持续时间: $BLOCK_DURATION_INT 秒"
    # 先调整块大小确保内容完整显示
    ffmpeg -y -loglevel error -i "$TEMP_DIR/blocks/block_0.png" \
      -vf "scale=$BACKGROUND_WIDTH:-1" "$TEMP_DIR/blocks/block_0_resized.png"
      
    ffmpeg -y -hwaccel cuda -loop 1 -i "$TEMP_DIR/blocks/block_0_resized.png" \
      -f lavfi -i "color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$BLOCK_DURATION_INT" \
      -filter_complex "
        [0:v]format=yuv420p,setpts=PTS-STARTPTS[img];
        [1:v]trim=duration=$BLOCK_DURATION_INT,setpts=PTS-STARTPTS[bg];
        [bg][img]overlay=0:-${EXPR1}:shortest=1
      " \
      -c:v h264_nvenc -preset p1 -r $FPS -t $BLOCK_DURATION_INT \
      "$TEMP_DIR/block_0_scroll.mp4"
    
    echo "file 'block_0_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    
    # 处理其余块
    for ((i=1; i<NUM_BLOCKS; i++)); do
      # 先调整块大小确保内容完整显示
      ffmpeg -y -loglevel error -i "$TEMP_DIR/blocks/block_$i.png" \
        -vf "scale=$BACKGROUND_WIDTH:-1" "$TEMP_DIR/blocks/block_${i}_resized.png"
      
      RESIZED_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$TEMP_DIR/blocks/block_${i}_resized.png")
      BLOCK_DURATION=$(echo "$RESIZED_HEIGHT / $SCROLL_SPEED" | bc -l)
      BLOCK_DURATION_INT=$(printf "%.0f" $BLOCK_DURATION)
      
      if [ $BLOCK_DURATION_INT -lt 1 ]; then
        BLOCK_DURATION_INT=1
      fi
      
      echo "处理块 $i，高度: $RESIZED_HEIGHT，持续时间: $BLOCK_DURATION_INT 秒"
      
      # 简化表达式，直接使用t乘以速度
      EXPR2="t*$SCROLL_SPEED"
      
      ffmpeg -y -hwaccel cuda -loop 1 -i "$TEMP_DIR/blocks/block_${i}_resized.png" \
        -f lavfi -i "color=white:s=${BACKGROUND_WIDTH}x${BACKGROUND_HEIGHT}:r=$FPS:d=$BLOCK_DURATION_INT" \
        -filter_complex "
          [0:v]format=yuv420p,setpts=PTS-STARTPTS[img];
          [1:v]trim=duration=$BLOCK_DURATION_INT,setpts=PTS-STARTPTS[bg];
          [bg][img]overlay=0:-${EXPR2}:shortest=1
        " \
        -c:v h264_nvenc -preset p1 -r $FPS -t $BLOCK_DURATION_INT \
        "$TEMP_DIR/block_${i}_scroll.mp4"
      
      echo "file 'block_${i}_scroll.mp4'" >> "$TEMP_DIR/concat_list.txt"
    done
    ;;
    
  *)
    echo "未知方法，使用默认scroll滤镜"
    METHOD=1
    ;;
esac

# 合并所有视频块
echo "合并视频块..."
cd "$TEMP_DIR"

# 直接使用concat demuxer合并视频，更快
ffmpeg -y -f concat -safe 0 -i "concat_list.txt" \
  -c copy "../$OUTPUT_VIDEO"

cd ..

RESULT=$?

PROCESS_END=$(date +%s.%N)
PROCESS_TIME=$(echo "$PROCESS_END - $PROCESS_START" | bc -l)
echo "滚动处理耗时: ${PROCESS_TIME} 秒"

# 计算总耗时
SCRIPT_END_TIME=$(date +%s.%N)
TOTAL_EXEC_TIME=$(echo "$SCRIPT_END_TIME - $SCRIPT_START_TIME" | bc -l)

if [ $RESULT -eq 0 ]; then
  # 性能统计
  echo ""
  echo "===== 性能统计 ====="
  echo "图像分析耗时: ${IMAGE_ANALYSIS_TIME} 秒"
  echo "帧切割耗时: ${FRAME_CUTTING_TIME} 秒 (占比: $(echo "scale=2; $FRAME_CUTTING_TIME * 100 / $TOTAL_EXEC_TIME" | bc -l)%)"
  echo "滚动处理耗时: ${PROCESS_TIME} 秒 (占比: $(echo "scale=2; $PROCESS_TIME * 100 / $TOTAL_EXEC_TIME" | bc -l)%)"
  echo "总执行时间: ${TOTAL_EXEC_TIME} 秒"
  echo "===================="
  
  echo "使用${METHOD_NAME}滚动视频处理成功！输出视频：$OUTPUT_VIDEO"
  
  # 清理临时文件
    rm -rf "$TEMP_DIR"
  
  exit 0
else
  echo "使用${METHOD_NAME}处理失败。错误代码：$RESULT"
  echo "总执行时间: ${TOTAL_EXEC_TIME} 秒"
  
  # 保留临时文件以便调试
  echo "临时文件保留在 $TEMP_DIR 目录"
  exit 1
fi 