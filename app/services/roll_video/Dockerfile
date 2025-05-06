###############################################################
# 第一阶段: 编译支持NVIDIA GPU的FFmpeg
###############################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS ffmpeg-builder

ENV DEBIAN_FRONTEND=noninteractive

# 修改镜像源并安装FFmpeg编译所需的最小依赖
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget pkg-config yasm nasm \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
    libva-dev libvdpau-dev git ca-certificates \
    zlib1g-dev libass-dev libmp3lame-dev libopus-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装nv-codec-headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout sdk/12.0 && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 编译FFmpeg
RUN mkdir -p /tmp/ffmpeg_sources && \
    cd /tmp/ffmpeg_sources && \
    wget -O ffmpeg-7.1.tar.gz https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz && \
    tar xvf ffmpeg-7.1.tar.gz && \
    cd ffmpeg-7.1 && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/cuda/lib64/pkgconfig" \
    ./configure \
    --prefix=/usr/local \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --enable-shared \
    --enable-gpl \
    --enable-nonfree \
    --enable-libfdk-aac \
    --enable-libx264 \
    --enable-libx265 \
    --enable-cuda \
    --enable-cuda-llvm \
    --enable-cuvid \
    --enable-nvenc \
    --enable-libnpp \
    --enable-pthreads \
    --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/lib" \
    --extra-libs="-lpthread -lm -lcuda -lcudart" && \
    make -j$(nproc) && \
    make install && \
    ldconfig

###############################################################
# 第二阶段: 安装cuDNN
###############################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cudnn-builder

RUN mkdir -p /cudnn-install
WORKDIR /cudnn-install

# 下载和安装cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget --no-check-certificate https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    mkdir -p /cudnn-extract && \
    tar -xf cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz -C /cudnn-extract && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###############################################################
# 第三阶段: Python依赖 (CPU版本)
###############################################################
FROM python:3.10-slim AS python-deps

ENV DEBIAN_FRONTEND=noninteractive

# 安装Python依赖
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple \
    ffmpeg-python==0.2.0 \
    opencv-python-headless==4.11.0.86 \
    numpy==1.26.4 \
    flask==3.1.0 \
    gunicorn==23.0.0 \
    pillow==10.4.0 \
    tqdm==4.67.1 \
    moviepy==2.1.2 \
    psutil==7.0.0 \
    && find /usr/local/lib/python3.*/site-packages/ -name "*.pyc" -delete \
    && find /usr/local/lib/python3.*/site-packages/ -name "__pycache__" -delete

###############################################################
# 最终阶段: 组装最终镜像
###############################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    FFMPEG_BINARY="/usr/local/bin/ffmpeg" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics \
    NVIDIA_REQUIRE_CUDA="cuda>=12.0"

WORKDIR /app

# 安装运行时必要的包，尽量精简
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    nvidia-driver-525-server \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 从cuDNN构建阶段复制文件
COPY --from=cudnn-builder /cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/
COPY --from=cudnn-builder /cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/

# 从FFmpeg构建阶段复制文件
COPY --from=ffmpeg-builder /usr/local/bin/ffmpeg /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/lib*so* /usr/local/lib/

# 从Python依赖阶段复制文件
COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/dist-packages

# 安装NVIDIA ML Python包和TensorFlow
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple -r dockerfile-requirements.txt \
    && find /usr/local/lib/python3.*/dist-packages/ -name "*.pyc" -delete \
    && find /usr/local/lib/python3.*/dist-packages/ -name "__pycache__" -delete

# 设置动态链接库配置
RUN ldconfig

# 验证FFmpeg是否支持NVIDIA编码器
# RUN ffmpeg -encoders | grep nvenc || echo "Warning: NVENC support not found"

# 暴露API端口
EXPOSE 5000

# 启动命令
# CMD ["python3", "/app/server/api_server.py"]
CMD ["tail", "-f", "/dev/null"]