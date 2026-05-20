# ─── Dockerfile ───────────────────────────────────────────────────
# 使用 CUDA 12.1 + Python 3.11 基底映像
# 注意：模型檔案（models/）不打包進 image，透過 volume 掛載
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 建立軟連結讓 python/pip 指令可用
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 設定工作目錄
WORKDIR /app

# 先複製 requirements.txt（利用 Docker layer cache）
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼（不含 models/ 和 data/，由 volume 掛載）
COPY agent/ ./agent/
COPY app/ ./app/
COPY scripts/ ./scripts/

# 建立必要資料夾
RUN mkdir -p data/raw data/processed models/classifier models/generator

# 開放 Streamlit 預設 port
EXPOSE 8501

# 健康檢查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 啟動指令
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]