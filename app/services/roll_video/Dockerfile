FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 非交互式
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    wget \
    cmake \
    git \
    clang \
    llvm \
    libtool \
    autoconf \
    automake \
    pkg-config \
    yasm \
    nasm \
    zlib1g-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libsdl2-dev \
    libass-dev \
    libva-dev \
    libvdpau-dev \
    libreadline-dev \
    libbz2-dev \
    libxext-dev \
    libgomp1 \
    liblzma-dev \
    libglib2.0-0 \
    mpich \
    openmpi-bin \
    libopenmpi-dev \
    gcc \
    g++ \
    make \
    zlib1g \
    openssl \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    gfortran \
    libblas3 \
    libopenblas-dev \
    libswresample-dev \
    libfreetype6-dev \
    libtheora-dev \
    libvorbis-dev \
    libmp3lame-dev \
    libopus-dev \
    # 安装 CUDA 开发工具
    cuda-nvcc-12-4 \
    cuda-nvrtc-12-4 \
    cuda-nvrtc-dev-12-4 \
    cuda-nvml-dev-12-4 \
    cuda-command-line-tools-12-4 \
    cuda-libraries-dev-12-4 \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装cuDNN
RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    tar -xf cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/ && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/ && \
    rm -rf /tmp/cudnn* && \
    ldconfig

# 下载并安装nv-codec-headers 注意显卡驱动版本
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout sdk/12.0 && \
    make -j$(nproc) && \
    make install

# 下载并编译支持NVIDIA硬件加速的FFmpeg
RUN cd /tmp && \
    wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz && \
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
    --enable-cuda-nvcc \
    --enable-libfreetype \
    --enable-libharfbuzz \
    --enable-libfontconfig \
    --enable-cuvid \
    --enable-nvenc \
    --enable-filter=scale_cuda \
    --enable-filter=scale_npp \
    --enable-filter=thumbnail_cuda \
    --enable-filter=overlay_cuda \
    --enable-filter=yadif_cuda \
    --enable-filter=bwdif_cuda \
    --enable-filter=scale \
    --enable-filter=crop \
    --enable-libnpp \
    --enable-pthreads \
    --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/lib" \
    --extra-libs="-lpthread -lm -lcuda -lcudart" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf ffmpeg* && \
    ldconfig

# 设置CUDA、FFmpeg环境变量
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    FFMPEG_NVENC_FLAGS="-I/usr/local/cuda/include" \
    FFMPEG_BINARY="/usr/local/bin/ffmpeg" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

# 验证FFmpeg是否支持NVIDIA编码器
# RUN ffmpeg -encoders | grep nvenc

# 复制
COPY dockerfile-requirements.txt /app/

# 设置工作目录
WORKDIR /app

# 安装Python依赖
RUN pip3 install --no-cache-dir -r dockerfile-requirements.txt

# 暴露API端口
EXPOSE 5000

# 启动命令
# CMD ["python3", "/app/server/api_server.py"]

CMD ["tail", "-f", "/dev/null"]
