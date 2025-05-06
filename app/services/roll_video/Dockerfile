###############################################################
# 第一阶段: 编译支持NVIDIA GPU的FFmpeg
###############################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS ffmpeg-builder

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    wget \
    cmake \
    git \
    vim \
    clang \
    llvm \
    clang-tools \
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
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    nvidia-driver-525-server \
    libreadline-dev \
    libbz2-dev \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgomp1 \
    liblzma-dev \
    libgl1-mesa-glx \
    libprotobuf-dev \
    protobuf-compiler \
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
    unzip \
    pciutils \
    net-tools \
    libblas-dev \
    gfortran \
    libblas3 \
    libopenblas-dev \
    git-lfs \
    libswresample-dev \
    libfreetype6-dev \
    libtheora-dev \
    libvorbis-dev \
    texinfo \
    libmp3lame-dev \
    libopus-dev \
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


# 安装nv-codec-headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout sdk/12.0 && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 安装 CUDA 开发工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-nvrtc-12-4 \
    cuda-nvrtc-dev-12-4 \
    cuda-nvml-dev-12-4 \
    cuda-command-line-tools-12-4 \
    cuda-libraries-dev-12-4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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
    --enable-cuvid \
    --enable-nvenc \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# 从FFmpeg构建阶段复制文件
COPY --from=ffmpeg-builder /cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/
COPY --from=ffmpeg-builder /cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/

COPY --from=ffmpeg-builder /usr/local/bin/ffmpeg /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/lib*so* /usr/local/lib/


# 安装NVIDIA ML Python包和TensorFlow
RUN pip3 install --no-cache-dir -r dockerfile-requirements.txt \
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
