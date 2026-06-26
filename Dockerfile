FROM m.daocloud.io/docker.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN sed -i \
      -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
      -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.txt ./
RUN pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.2.0" \
    && pip install \
      --index-url https://mirrors.aliyun.com/pypi/simple/ \
      -r requirements-docker.txt \
    && pip install \
      --index-url https://mirrors.aliyun.com/pypi/simple/ \
      --no-deps \
      "openai-whisper>=20231117"

RUN pip install \
      --index-url https://mirrors.aliyun.com/pypi/simple/ \
      "tqdm>=4.66.0" \
      "scikit-learn==1.8.0"

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
