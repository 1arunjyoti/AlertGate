FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libjpeg62-turbo \
        libturbojpeg0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch wheels first so the image avoids CUDA packages.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu \
    && grep -viE '^(torch|torchvision|pyreadline3)==' requirements.txt > /tmp/requirements-docker.txt \
    && pip install -r /tmp/requirements-docker.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY config ./config
COPY src ./src
COPY [Rr][Ee][Aa][Dd][Mm][Ee].md LICENSE ./

RUN mkdir -p media/snapshots media/clips logs /tmp/matplotlib /tmp/ultralytics \
    && useradd --create-home --shell /usr/sbin/nologin alertgate \
    && chown -R alertgate:alertgate /app /tmp/matplotlib /tmp/ultralytics

USER alertgate

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/stats >/dev/null || exit 1

CMD ["python", "src/main.py"]
