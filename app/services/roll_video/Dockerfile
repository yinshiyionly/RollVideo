FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 设置非交互式环境和工作目录
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    FFMPEG_NVENC_FLAGS="-I/usr/local/cuda/include" \
    FFMPEG_BINARY="/usr/local/bin/ffmpeg" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics \
    NVIDIA_REQUIRE_CUDA="cuda>=12.0"

WORKDIR /app

# 修改镜像源并安装所有必要的系统依赖、CUDA开发工具
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-pip wget cmake git clang llvm clang-tools \
    libtool autoconf automake pkg-config yasm nasm zlib1g-dev libx264-dev libx265-dev \
    libvpx-dev libfdk-aac-dev libsdl2-dev libass-dev libva-dev libvdpau-dev \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev nvidia-driver-525-server \
    libreadline-dev libbz2-dev libsm6 libxrender1 libxext-dev libgomp1 \
    liblzma-dev libgl1-mesa-glx libprotobuf-dev protobuf-compiler libglib2.0-0 \
    mpich openmpi-bin libopenmpi-dev gcc g++ make zlib1g openssl libsqlite3-dev \
    libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 \
    libopenblas-dev libswresample-dev libfreetype6-dev libtheora-dev libvorbis-dev \
    texinfo libmp3lame-dev libopus-dev cuda-nvcc-12-4 cuda-nvrtc-12-4 \
    cuda-nvrtc-dev-12-4 cuda-nvml-dev-12-4 cuda-command-line-tools-12-4 \
    cuda-libraries-dev-12-4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装cuDNN、nv-codec-headers和FFmpeg(支持NVIDIA硬件加速)
RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    tar -xf cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/* /usr/local/cuda/lib64/ && \
    cp -r cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/* /usr/local/cuda/include/ && \
    rm -rf /tmp/cudnn* && \
    ldconfig && \
    # 安装nv-codec-headers
    git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout sdk/12.0 && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    # 安装FFmpeg
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
    rm -rf ffmpeg* nv-codec-headers && \
    ldconfig && \
    # 验证FFmpeg是否支持NVIDIA编码器
    ffmpeg -encoders | grep nvenc

# 安装Python依赖
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple \
    ffmpeg-python==0.2.0 \
    opencv-python==4.11.0.86 \
    numpy==1.26.4 \
    flask==3.1.0 \
    gunicorn==23.0.0 \
    tensorflow==2.15.0 \
    pillow==10.4.0 \
    tqdm==4.67.1 \
    moviepy==2.1.2 \
    psutil==7.0.0 \
    nvidia-ml-py3==7.352.0

# 暴露API端口
EXPOSE 5000

# 启动命令
# 5. 用 supervisor 启动主程序和 celery 队列
# CMD ["supervisord", "-c", "/roll-video/supervisor-conf/supervisord.conf"]

# 启动命令
# CMD ["python3", "/app/server/api_server.py"]
CMD ["tail", "-f", "/dev/null"]