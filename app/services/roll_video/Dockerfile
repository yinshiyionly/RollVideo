###############################################################
# 第一阶段: 编译支持NVIDIA GPU的FFmpeg
###############################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS ffmpeg-builder

# 非交互式
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖，优化包安装和清理过程
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    wget \
    cmake \
    git \
    clang \
    yasm \
    nasm \
    pkg-config \
    # FFmpeg 必需的开发库
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libass-dev \
    libva-dev \
    libvdpau-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    nvidia-driver-525-server \
    # 其他必要的库
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgomp1 \
    libgl1-mesa-glx \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# 创建工作目录
WORKDIR /build

# 安装cuDNN，优化下载和清理过程
RUN wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz \
    && mkdir -p cudnn-extract \
    && tar -xf cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz -C cudnn-extract \
    && cp -r cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/ \
    && cp -r cudnn-extract/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/ \
    && rm -rf cudnn-extract cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz \
    && ldconfig

# 安装nv-codec-headers
RUN git clone --depth 1 -b sdk/12.0 https://github.com/FFmpeg/nv-codec-headers.git \
    && cd nv-codec-headers \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf nv-codec-headers

# 安装 CUDA 开发工具，只安装必要的包
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-nvrtc-12-4 \
    cuda-nvrtc-dev-12-4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# 下载并编译FFmpeg，优化编译过程
RUN wget -q https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz \
    && tar xf ffmpeg-7.1.tar.gz \
    && cd ffmpeg-7.1 \
    && PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/cuda/lib64/pkgconfig" \
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
    --extra-libs="-lpthread -lm -lcuda -lcudart" \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ffmpeg* \
    && ldconfig

###############################################################
# 最终阶段: 组装最终镜像
###############################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    FFMPEG_BINARY="/usr/local/bin/ffmpeg" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

WORKDIR /app

# 安装运行时必要的包
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    supervisor \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    nvidia-driver-525-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# 从构建阶段复制必要的文件
COPY --from=ffmpeg-builder /usr/local/cuda/lib64/libcudnn* /usr/local/cuda/lib64/
COPY --from=ffmpeg-builder /usr/local/cuda/include/cudnn*.h /usr/local/cuda/include/
COPY --from=ffmpeg-builder /usr/local/bin/ffmpeg /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/lib*so* /usr/local/lib/

# 复制应用代码和依赖文件
COPY dockerfile-requirements.txt .

# 安装 Python 包并清理缓存
RUN pip3 install --no-cache-dir -r dockerfile-requirements.txt \
    && find /usr/local/lib/python3.*/dist-packages/ -name "*.pyc" -delete \
    && find /usr/local/lib/python3.*/dist-packages/ -name "__pycache__" -delete \
    && rm -rf ~/.cache/pip/*

# 设置动态链接库配置
RUN ldconfig

EXPOSE 5000

CMD ["tail", "-f", "/dev/null"]
