# 使用Python 3.11 本地已经有
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 完全替换为阿里云镜像源（确保匹配bookworm版本）
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 安装系统依赖，添加--fix-missing参数增强稳定性
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 配置pip使用阿里云镜像
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# 复制requirements.txt文件
COPY requirements.txt .

# 安装核心Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 1124

# 启动应用
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1124"]
