### ===== 阶段 1：构建带 CUDA 的 FFmpeg =====
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS ffmpeg-builder

# 设置非交互式环境和工作目录
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config yasm nasm wget git cmake \
    libx264-dev libx265-dev libfdk-aac-dev libvpx-dev libass-dev libopus-dev \
    libmp3lame-dev libvorbis-dev libfreetype6-dev libtheora-dev \
    zlib1g-dev cuda-nvcc-12-4 cuda-nvrtc-12-4 cuda-nvrtc-dev-12-4 \
    cuda-nvml-dev-12-4 cuda-command-line-tools-12-4 cuda-libraries-dev-12-4 \
    ca-certificates && \
    # 安装 cuDNN
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    tar -xf cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/ && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/ && \
    rm -rf cudnn* && \
    ldconfig && \
    # 编译 nv-codec-headers
    git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && git checkout sdk/12.0 && make && make install && \
    cd /build && \
    # 下载并构建 FFmpeg
    wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz && \
    tar -xf ffmpeg-7.1.tar.gz && cd ffmpeg-7.1 && \
    ./configure \
      --prefix=/ffmpeg-build \
      --disable-debug \
      --disable-doc \
      --disable-ffplay \
      --enable-shared \
      --enable-gpl --enable-nonfree \
      --enable-cuda --enable-cuvid --enable-nvenc --enable-libnpp \
      --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libvpx \
      --enable-libmp3lame --enable-libopus --enable-libvorbis \
      --enable-libfreetype --enable-libass \
      --extra-cflags="-I/usr/local/cuda/include" \
      --extra-ldflags="-L/usr/local/cuda/lib64" && \
    make -j$(nproc) && make install

### ===== 阶段 2：运行环境 + Python 服务端 =====
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

WORKDIR /app

# 安装运行环境依赖和 Python 3
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender1 wget curl unzip pciutils net-tools supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 拷贝编译好的 FFmpeg 二进制
COPY --from=ffmpeg-builder /ffmpeg-build/ /usr/local/

# 安装 Python 依赖（注意已用阿里云源）
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple -r dockerfile-requirements.txt

# 暴露API端口
EXPOSE 5000

# 启动命令
# 5. 用 supervisor 启动主程序和 celery 队列
# CMD ["supervisord", "-c", "/roll-video/supervisor-conf/supervisord.conf"]

# 启动命令
# CMD ["python3", "/app/server/api_server.py"]
CMD ["tail", "-f", "/dev/null"]