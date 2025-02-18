# 使用 Python 3.11 slim 版本作为基础镜像
# slim版本是一个轻量级的Python镜像，只包含运行Python应用所必需的组件
# 相比完整版镜像体积更小，更适合部署生产环境
FROM python:3.11-slim

# 设置工作目录
# 在容器内创建/app目录并将其设置为工作目录
# 后续的操作（如COPY）如果使用相对路径，都会基于这个目录
WORKDIR /app

# 设置环境变量
# PYTHONUNBUFFERED=1：确保Python的输出不会被缓存，实时输出日志
# PYTHONDONTWRITEBYTECODE=1：防止Python将pyc文件写入磁盘
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装项目依赖包
# --no-cache-dir：不缓存pip下载的包，减少镜像大小
# 指定精确的版本号以确保构建的一致性和可重现性
# aiohttp: 用于异步HTTP请求
# colorlog: 用于彩色日志输出
# fastapi: Web框架
# python-dotenv: 用于加载.env环境变量
# tiktoken: OpenAI的分词器
# uvicorn: ASGI服务器
RUN pip install --no-cache-dir \
    aiohttp==3.11.11 \
    colorlog==6.9.0 \
    fastapi==0.115.8 \
    python-dotenv==1.0.1 \
    tiktoken==0.8.0 \
    "uvicorn[standard]"

# 复制项目文件
# 将本地的./app目录下的所有文件复制到容器中的/app/app目录
COPY ./app ./app

# 暴露端口
# 声明容器将使用1124端口
EXPOSE 1124

# python -m uvicorn：通过Python模块的方式启动uvicorn服务器
# app.main:app：指定FastAPI应用的导入路径，格式为"模块路径:应用实例变量名"
# --host 0.0.0.0：允许来自任何IP的访问（不仅仅是localhost）
# --port 1124：指定服务器监听的端口号
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1124"]
