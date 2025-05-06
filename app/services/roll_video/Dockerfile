# app/services/roll_video/Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}" \
    FFMPEG_BINARY="/usr/local/bin/ffmpeg" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-setuptools \
    libsm6 libxext6 libxrender-dev libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 注入 FFmpeg 二进制（由上一阶段构建）
COPY ffmpeg-bin/ffmpeg /usr/local/bin/ffmpeg
COPY ffmpeg-bin/ffprobe /usr/local/bin/ffprobe
COPY ffmpeg-bin/lib*so* /usr/local/lib/

RUN ldconfig

COPY . /app

RUN pip3 install --no-cache-dir -r dockerfile-requirements.txt

EXPOSE 5000

CMD ["tail", "-f", "/dev/null"]