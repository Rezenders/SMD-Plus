FROM ultralytics/ultralytics:latest

RUN apt update && apt install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-pandas \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*