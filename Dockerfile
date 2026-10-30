FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
 && python3.11 -m pip install --no-cache-dir -r requirements.txt
COPY bai ./bai
ENTRYPOINT ["python3.11", "-m", "bai.analyze"]
