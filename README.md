<div>
<h1>DeepClaude 🐬🧠 - OpenAI Compatible</h1>

<a href="https://github.com/getasterisk/deepclaude"> Inspiration from getasterisk/deepclaude</a>

[![GitHub license](https://img.erlich.fun/personal-blog/uPic/deepclaude.svg)](#)
[![Compatible with](https://img.shields.io/badge/-ChatGPT-412991?style=flat-square&logo=openai&logoColor=FFFFFF)](https://openai.com)

</div>

<div>
<h3 style="color: #FF9909"> 特别说明：对于不太会部署，只是希望使用上最强 DeepClaude 组合的朋友，可以直接访问 kakadeai 个人网站自助购买按量付费的 API：https://erlich.fun/deepclaude-pricing
也可以直接联系 kakadeai国内可以直接访问 </h3>
</div>

---

<details>
<summary><strong>更新日志：</strong></summary> 
<div>
2025-02-08.2: 支持非流式请求，支持 OpenAI 兼容的 models 接口返回。（⚠️ 当前暂未实现正确的 tokens 消耗统计，稍后更新）

2025-02-08.1: 添加 Github Actions，支持 fork 自动同步、支持自动构建 Docker 最新镜像、支持 docker-compose 部署

2025-02-07.2: 修复 Claude temperature 参数可能会超过范围导致的请求失败的 bug

2025-02-07.1: 支持 Claude temputerature 等参数；添加更详细的 .env.example 说明

2025-02-06.1：修复非原生推理模型无法获得到推理内容的 bug

2025-02-05.1: 支持通过环境变量配置是否是原生支持推理字段的模型，满血版本通常支持

2025-02-04.2: 支持跨域配置，可在 .env 中配置

2025-02-04.1: 支持 Openrouter 以及 OneAPI 等中转服务商作为 Claude 部分的供应商

2025-02-03.3: 支持 OpenRouter 作为 Claude 的供应商，详见 .env.example 说明

2025-02-03.2: 由于 deepseek r1 在某种程度上已经开启了一个规范，所以我们也遵循推理标注的这种规范，更好适配支持的更好的 Cherry Studio 等软件。

2025-02-03.1: Siliconflow 的 DeepSeek R1 返回结构变更，支持新的返回结构

</div>
</details>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Implementation](#implementation)
- [How to run](#how-to-run)
  - [1. 获得运行所需的 API](#1-获得运行所需的-api)
  - [2. 开始运行（本地运行）](#2-开始运行本地运行)
- [Deployment](#deployment)
  - [Railway 一键部署（推荐）](#railway-一键部署推荐)
  - [Zeabur 一键部署(一定概率下会遇到 Domain 生成问题，需要重新创建 project 部署)](#zeabur-一键部署一定概率下会遇到-domain-生成问题需要重新创建-project-部署)
  - [使用 docker-compose 部署（Docker 镜像将随着 main 分支自动更新到最新）](#使用-docker-compose-部署docker-镜像将随着-main-分支自动更新到最新)
  - [Docker 部署（自行 Build）](#docker-部署自行-build)
- [Automatic fork sync](#automatic-fork-sync)
- [Technology Stack](#technology-stack)
- [Star History](#star-history)
- [Buy me a coffee](#buy-me-a-coffee)
- [About Me](#about-me)

# Introduction
最近 DeepSeek 推出了 [DeepSeek R1 模型](https://platform.deepseek.com)，在推理能力上已经达到了第一梯队。但是 DeepSeek R1 在一些日常任务的输出上可能仍然无法匹敌 Claude 3.5 Sonnet。Aider 团队最近有一篇研究，表示通过[采用 DeepSeek R1 + Claude 3.5 Sonnet 可以实现最好的效果](https://aider.chat/2025/01/24/r1-sonnet.html)。

<img src="https://img.erlich.fun/personal-blog/uPic/heiQYX.png" alt="deepseek r1 and sonnet benchmark" style="width=400px;"/>

> **R1 as architect with Sonnet as editor has set a new SOTA of 64.0%** on the [aider polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html). They achieve this at **14X less cost** compared to the previous o1 SOTA result.

并且 Aider 还 [开源了 Demo](https://github.com/getasterisk/deepclaude)，你可以直接在他们的项目上进行在线体验。



本项目受到该项目的启发，通过 fastAPI 完全重写，并支持 OpenAI 兼容格式，支持 DeepSeek 官方 API 以及第三方托管的 API。

用户可以自行运行在自己的服务器，并对外提供开放 API 接口，接入 [OneAPI](https://github.com/songquanpeng/one-api) 等实现统一分发（token 消耗部分仍需开发）。也可以接入你的日常 ChatBox  软件以及 接入 [Cursor](https://www.cursor.com/) 等软件实现更好的编程效果（Claude 的流式输出+ Tool use 仍需开发）。

# Implementation
⚠️Notice: 目前只支持流式输出模式（因为这是效率最高的模式，不会浪费时间）；接下来会实现第一段 DeepSeek 推理阶段流式，Claude 输出非流式的模式（处于节省时间的考虑）。

![image-20250201212456050](https://img.erlich.fun/personal-blog/uPic/image-20250201212456050.png)

# How to run

> 项目支持本地运行和服务器运行，本地运行可与 Ollama 搭配，实现用本地的 DeepSeek R1 与 Claude 组合输出


## 1. 获得运行所需的 API

1. 获取 DeepSeek API，因为最近 DeepSeek 还在遭受攻击，所以经常无法使用，推荐使用 Siliconflow 的效果更好（也可以本地 Ollama 的）: https://cloud.siliconflow.cn/i/RXikvHE2 (点击此链接可以获得到 2000 万免费 tokens)
2. 获取 Claude 的 API KEY （目前还没有做中转模式，以及对 Google 和 AWS 托管的版本的兼容支持，欢迎 PR）：https://console.anthropic.com

## 2. 开始运行（本地运行）

### 1. 安装 uv 包管理器

```bash
# 使用 pip 安装 uv
pip install uv

# 或使用 curl 安装（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 项目依赖安装

```bash
# 克隆项目
git clone https://github.com/ErlichLiu/DeepClaude.git
cd DeepClaude

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. 环境变量配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 .env 文件，配置必要的环境变量：
```env
# API 访问控制
ALLOW_API_KEY=your_allow_api_key  # 设置访问API的密钥
ALLOW_ORIGINS="*"                # 允许的跨域来源

# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner
IS_ORIGIN_REASONING=true

# Claude API 配置
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_PROVIDER=anthropic
CLAUDE_API_URL=https://api.anthropic.com/v1/messages

# 日志配置
LOG_LEVEL=INFO  # 可选：DEBUG, INFO, WARNING, ERROR
```

### 4. 启动本地服务

```bash
# 基本启动
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Step 6. 配置程序到你的 Chatbox

以下是几个推荐的Chatbox及其配置方法：

### NextChat

1. 访问 [NextChat](https://nextchat.dev/) 并登录
2. 点击左下角的设置图标
3. 在「API设置」中选择「自定义API」
4. 填写以下信息：
   - API Key: 你在.env中设置的ALLOW_API_KEY
   - Base URL: http://127.0.0.1:8000/v1（本地部署）或你的服务器地址
   - 模型: deepclaude
5. 点击保存即可开始对话

### ChatBox

1. 下载并安装 [ChatBox](https://chatboxai.app/zh)
2. 点击左侧「设置」-「API设置」
3. 添加自定义API配置：
   - 名称：DeepClaude
   - Base URL：http://127.0.0.1:8000/v1
   - API Key：你的ALLOW_API_KEY
   - 模型：deepclaude
4. 保存配置后即可在对话中选择DeepClaude模型

### LobeChat

1. 访问 [LobeChat](https://lobechat.com/) 或部署自己的实例
2. 进入设置页面
3. 选择「Language Model」-「Add Custom Model」
4. 填写配置：
   - 名称：DeepClaude
   - Endpoint：http://127.0.0.1:8000/v1
   - API Key：你的ALLOW_API_KEY
   - 模型：deepclaude
5. 保存后即可在会话中使用DeepClaude模型

> 注意：如果是服务器部署，请将http://127.0.0.1:8000替换为你的服务器地址

**注：本项目采用 uv 作为包管理器，这是一个更快速更现代的管理方式，用于替代 pip，你可以[在此了解更多](https://docs.astral.sh/uv/)**



# Deployment

> 项目支持 Docker 服务器部署，可自行调用接入常用的 Chatbox，也可以作为渠道一直，将其视为一个特殊的 `DeepClaude`模型接入到 [OneAPI](https://github.com/songquanpeng/one-api) 等产品使用。

## 使用 docker-compose 部署（推荐）

1. 确保已安装 Docker 和 Docker Compose

2. 创建 docker-compose.yml：
```yaml
services:
  deepclaude:
    image: ghcr.io/erlichliu/deepclaude:latest
    ports:
      - "8000:8000"
    environment:
      ALLOW_API_KEY: your_allow_api_key
      ALLOW_ORIGINS: "*"
      DEEPSEEK_API_KEY: your_deepseek_api_key
      DEEPSEEK_API_URL: https://api.deepseek.com/v1/chat/completions
      DEEPSEEK_MODEL: deepseek-reasoner
      IS_ORIGIN_REASONING: true
      CLAUDE_API_KEY: your_claude_api_key
      CLAUDE_MODEL: claude-3-5-sonnet-20241022
      CLAUDE_PROVIDER: anthropic
      CLAUDE_API_URL: https://api.anthropic.com/v1/messages
      LOG_LEVEL: INFO
    restart: always
```

3. 启动服务：
```bash
docker-compose up -d
```

## 手动构建Docker镜像

```bash
# 构建镜像
docker build -t deepclaude:latest .

# 运行容器
docker run -d \
    -p 8000:8000 \
    -e ALLOW_API_KEY=your_allow_api_key \
    -e ALLOW_ORIGINS="*" \
    -e DEEPSEEK_API_KEY=your_deepseek_api_key \
    -e DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions \
    -e DEEPSEEK_MODEL=deepseek-reasoner \
    -e IS_ORIGIN_REASONING=true \
    -e CLAUDE_API_KEY=your_claude_api_key \
    -e CLAUDE_MODEL=claude-3-5-sonnet-20241022 \
    -e CLAUDE_PROVIDER=anthropic \
    -e CLAUDE_API_URL=https://api.anthropic.com/v1/messages \
    -e LOG_LEVEL=INFO \
    --restart always \
    deepclaude:latest
```

## 服务器部署配置

### Nginx反向代理配置

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 安全配置建议

1. **API访问控制**
- 设置ALLOW_API_KEY进行认证
- 配置ALLOW_ORIGINS限制跨域访问

2. **SSL/TLS配置**
- 使用Let's Encrypt配置HTTPS
- 启用SSL证书自动更新

3. **Docker安全配置**
- 限制容器资源使用
- 配置容器网络隔离
- 定期更新镜像

### 监控和日志

1. **日志配置**
- 设置LOG_LEVEL控制日志级别
- 配置日志轮转策略

2. **监控指标**
- API请求量监控
- 响应时间监控
- 错误率监控
- 资源使用监控

```bash
# 基本启动
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Step 6. 配置程序到你的 Chatbox

以下是几个推荐的Chatbox及其配置方法：

### NextChat

1. 访问 [NextChat](https://nextchat.dev/) 并登录
2. 点击左下角的设置图标
3. 在「API设置」中选择「自定义API」
4. 填写以下信息：
   - API Key: 你在.env中设置的ALLOW_API_KEY
   - Base URL: http://127.0.0.1:8000/v1（本地部署）或你的服务器地址
   - 模型: deepclaude
5. 点击保存即可开始对话

### ChatBox

1. 下载并安装 [ChatBox](https://chatboxai.app/zh)
2. 点击左侧「设置」-「API设置」
3. 添加自定义API配置：
   - 名称：DeepClaude
   - Base URL：http://127.0.0.1:8000/v1
   - API Key：你的ALLOW_API_KEY
   - 模型：deepclaude
4. 保存配置后即可在对话中选择DeepClaude模型

### LobeChat

1. 访问 [LobeChat](https://lobechat.com/) 或部署自己的实例
2. 进入设置页面
3. 选择「Language Model」-「Add Custom Model」
4. 填写配置：
   - 名称：DeepClaude
   - Endpoint：http://127.0.0.1:8000/v1
   - API Key：你的ALLOW_API_KEY
   - 模型：deepclaude
5. 保存后即可在会话中使用DeepClaude模型

> 注意：如果是服务器部署，请将http://127.0.0.1:8000替换为你的服务器地址

**注：本项目采用 uv 作为包管理器，这是一个更快速更现代的管理方式，用于替代 pip，你可以[在此了解更多](https://docs.astral.sh/uv/)**



# Deployment

> 项目支持 Docker 服务器部署，可自行调用接入常用的 Chatbox，也可以作为渠道一直，将其视为一个特殊的 `DeepClaude`模型接入到 [OneAPI](https://github.com/songquanpeng/one-api) 等产品使用。

## 腾讯云ECS部署指南

### 1. 环境准备

1. 登录腾讯云控制台，创建Ubuntu实例（推荐Ubuntu 20.04 LTS）
2. 配置安全组，开放8000端口
3. 使用SSH连接到服务器

### 2. 安装Docker环境

```bash
# 更新包索引
sudo apt-get update

# 安装必要的系统工具
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# 添加Docker官方GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 添加Docker软件源
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 再次更新包索引
sudo apt-get update

# 安装Docker CE
sudo apt-get install -y docker-ce

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组（可选，需要重新登录生效）
sudo usermod -aG docker $USER
```

### 3. 部署DeepClaude

1. 创建项目目录并进入：
```bash
mkdir deepclaude && cd deepclaude
```

2. 创建docker-compose.yml文件：
```bash
cat > docker-compose.yml << 'EOF'
version: '3'
services:
  deepclaude:
    image: erlichliu/deepclaude:latest
    container_name: deepclaude
    ports:
      - "8000:8000"
    environment:
      - ALLOW_API_KEY=your_allow_api_key
      - ALLOW_ORIGINS="*"
      - DEEPSEEK_API_KEY=your_deepseek_api_key
      - DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
      - DEEPSEEK_MODEL=deepseek-reasoner
      - IS_ORIGIN_REASONING=true
      - CLAUDE_API_KEY=your_claude_api_key
      - CLAUDE_MODEL=claude-3-5-sonnet-20241022
      - CLAUDE_PROVIDER=anthropic
      - CLAUDE_API_URL=https://api.anthropic.com/v1/messages
      - LOG_LEVEL=INFO
    restart: always
EOF
```

3. 修改配置：
使用vim或其他编辑器修改docker-compose.yml中的环境变量，替换your_allow_api_key、your_deepseek_api_key和your_claude_api_key为实际的值。

4. 启动服务：
```bash
docker-compose up -d
```

5. 检查服务状态：
```bash
docker-compose ps
docker-compose logs
```

### 4. 配置域名和SSL（可选）

1. 在腾讯云购买域名并完成备案
2. 添加DNS解析记录，将域名指向ECS服务器IP
3. 安装Nginx：
```bash
sudo apt-get install -y nginx
```

4. 配置Nginx反向代理：
```bash
sudo vim /etc/nginx/sites-available/deepclaude
```

添加以下配置：
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

5. 启用站点配置：
```bash
sudo ln -s /etc/nginx/sites-available/deepclaude /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

6. 安装SSL证书：
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

完成以上步骤后，你就可以通过 https://your_domain.com 安全地访问你的DeepClaude API了。

### 5. 常见问题排查

1. 如果服务无法启动，检查docker-compose.yml中的环境变量配置
2. 如果无法访问API，检查安全组配置和防火墙设置
3. 查看服务日志：
```bash
docker-compose logs -f
```

4. 重启服务：
```bash
docker-compose restart
```

## Railway 一键部署（推荐）
<details>
<summary><strong>一键部署到 Railway</strong></summary> 

<div>
1. 首先 fork 一份代码。

2. 点击打开 Railway 主页：https://railway.com?referralCode=RNTGCA
   
3. 点击 `Deploy a new project`
![image-20250209164454358](https://img.erlich.fun/personal-blog/uPic/image-20250209164454358.png)

4. 点击 `Deploy from GitHub repo`
![image-20250209164638713](https://img.erlich.fun/personal-blog/uPic/image-20250209164638713.png)

5. 点击 `Login with GitHub`
![image-20250209164843566](https://img.erlich.fun/personal-blog/uPic/image-20250209164843566.png)

6. 选择升级，选择只需 5 美金的 Hobby Plan 即可 
![image-20250209165034070](https://img.erlich.fun/personal-blog/uPic/image-20250209165034070.png)
![image-20250209165108355](https://img.erlich.fun/personal-blog/uPic/image-20250209165108355.png)

1. 点击 `Create a New Project`
![create-a-new-project](https://img.erlich.fun/personal-blog/uPic/rvfGTE.png)

1. 继续选择 `Deploy from GitHub repo`
![image-20250209164638713](https://img.erlich.fun/personal-blog/uPic/image-20250209164638713.png)

1. 输入框内搜索`DeepClaude`，选中后点击。
![deploy-from-github-repo](https://img.erlich.fun/personal-blog/uPic/ihOzXU.png)

1.  选择`Variable`，并点击`New Variable` 按钮，按照环境变量内的键值对进行填写
![variable](https://img.erlich.fun/personal-blog/uPic/VrZgxp.png)

1.  填写完成后重新点击 `Deploy` 按钮，等待数秒后即可完成部署
![deploy](https://img.erlich.fun/personal-blog/uPic/5kvkLI.png)

1.  部署完成后，点击 `Settings` 按钮，然后向下查看到 `Networking` 区域，然后选择 `Generate Domain`，并输入 `8000` 作为端口号
![networking](https://img.erlich.fun/personal-blog/uPic/PQyAtG.png)
![generate-domain](https://img.erlich.fun/personal-blog/uPic/i5JnX8.png)
![port](https://img.erlich.fun/personal-blog/uPic/ZEwxRm.png)

1.  接下来就可以在你喜欢的 Chatbox 内配置使用或作为 API 使用了
![using](https://img.erlich.fun/personal-blog/uPic/hD8V6e.png)

</div>
</details>

## Zeabur 一键部署(一定概率下会遇到 Domain 生成问题，需要重新创建 project 部署)
<details>
<summary><strong>一键部署到 Zeabur</strong></summary> 
<div>


[![Deployed on Zeabur](https://zeabur.com/deployed-on-zeabur-dark.svg)](https://zeabur.com?referralCode=ErlichLiu&utm_source=ErlichLiu)

 1. 首先 fork 一份代码。
 2. 进入 [Zeabur](https://zeabur.com?referralCode=ErlichLiu&utm_source=ErlichLiu)，登录。
 3. 选择 Create New Project，选择地区为新加坡或日本区域。
 4. 选择项目来源为 Github，搜索框搜索 DeepClaude 后确认，然后点击右下角的 Config。
 5. 在 Environment Variables 区域点击 Add Environment Variables，逐个填写 .env.example 当中的配置，等号左右对应的就是 Environment Variables 里的 Key 和 Value。（注意：ALLOW_API_KEY 是你自己规定的外部访问你的服务时需要填写的 API KEY，可以随意填写，不要有空格）
 6. 全部编辑完成后点击 Next，然后点击 Deploy，静待片刻即可完成部署。
 7. 完成部署后点击当前面板上部的 Networking，点击 Public 区域的 Generate Domain（也可以配置自己的域名），然后输入一个你想要的域名即可（这个完整的 xxx.zeabur.app 将是你接下来在任何开源对话框、Cursor、Roo Code 等产品内填写的 baseUrl）
 8. 接下来就可以去上述所说的任何的项目里去配置使用你的 API 了，也可以配置到 One API，作为一个 OpenAI 渠道使用。（晚点会补充这部分的配置方法）
</div>
</details>

## 使用 docker-compose 部署（Docker 镜像将随着 main 分支自动更新到最新）

   推荐可以使用 `docker-compose.yml` 文件进行部署，更加方便快捷。

   1. 确保已安装 Docker Compose。
   2. 复制 `docker-compose.yml` 文件到项目根目录。
   3. 修改 `docker-compose.yml` 文件中的环境变量配置，将 `your_allow_api_key`，`your_allow_origins`，`your_deepseek_api_key` 和 `your_claude_api_key` 替换为你的实际配置。
   4. 在项目根目录下运行 Docker Compose 命令启动服务：

      ```bash
      docker-compose up -d
      ```

   服务启动后，DeepClaude API 将在 `http://宿主机IP:8000/v1/chat/completions` 上进行访问。


## Docker 部署（自行 Build）

1. **构建 Docker 镜像**

   在项目根目录下，使用 Dockerfile 构建镜像。请确保已经安装 Docker 环境。

   ```bash
   docker build -t deepclaude:latest .
   ```

2. **运行 Docker 容器**

   运行构建好的 Docker 镜像，将容器的 8000 端口映射到宿主机的 8000 端口。同时，通过 `-e` 参数设置必要的环境变量，包括 API 密钥、允许的域名等。请根据 `.env.example` 文件中的说明配置环境变量。

   ```bash
   docker run -d \
       -p 8000:8000 \
       -e ALLOW_API_KEY=your_allow_api_key \
       -e ALLOW_ORIGINS="*" \
       -e DEEPSEEK_API_KEY=your_deepseek_api_key \
       -e DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions \
       -e DEEPSEEK_MODEL=deepseek-reasoner \
       -e IS_ORIGIN_REASONING=true \
       -e CLAUDE_API_KEY=your_claude_api_key \
       -e CLAUDE_MODEL=claude-3-5-sonnet-20241022 \
       -e CLAUDE_PROVIDER=anthropic \
       -e CLAUDE_API_URL=https://api.anthropic.com/v1/messages \
       -e LOG_LEVEL=INFO \
       --restart always \
       deepclaude:latest
   ```

   请替换上述命令中的 `your_allow_api_key`，`your_allow_origins`，`your_deepseek_api_key` 和 `your_claude_api_key` 为你实际的 API 密钥和配置。`ALLOW_ORIGINS` 请设置为允许访问的域名，如 `"http://localhost:3000,https://chat.example.com"` 或 `"*"` 表示允许所有来源。

## Dify 部署方案

### 1. 系统要求

- Docker Engine 20.10.0+
- Docker Compose V2+
- 最小配置：2核CPU、8GB内存、20GB存储

### 2. 快速部署

```bash
# 克隆仓库
git clone https://github.com/langgenius/dify.git
cd dify/docker

# 配置环境
cp .env.example .env

# 编辑 .env 文件，设置必要参数
# - 核心配置（CONSOLE_URL, API_URL等）
# - 数据库配置（DB_HOST, DB_PASSWORD等）
# - Redis配置（REDIS_HOST, REDIS_PASSWORD等）
# - LLM配置（OPENAI_API_KEY等）

# 启动服务
docker compose pull
docker compose up -d

# 初始化数据库
docker compose exec api flask db upgrade
docker compose exec api flask create-admin
```

### 3. 访问服务

- 控制台：http://localhost:3000
- API服务：http://localhost:5001

更多详细配置和自定义模型集成说明，请参考 [项目实施部署指南](doc/项目实施部署.md#dify-部署方案)

# Automatic fork sync
项目已经支持 Github Actions 自动更新 fork 项目的代码，保持你的 fork 版本与当前 main 分支保持一致。如需开启，请 frok 后在 Settings 中开启 Actions 权限即可。


# Technology Stack

## 系统架构

### 三层架构
- 入口层 (app/main.py)
  - 系统入口和API路由
  - 环境配置加载
  - 中间件处理
  - 跨域配置

- 核心业务层 (app/deepclaude/)
  - 实现核心业务逻辑
  - 消息处理和转发
  - 流式响应处理
  - 错误处理和重试机制

- 客户端层 (app/clients/)
  - API客户端封装
  - 请求格式转换
  - 响应处理
  - 异常处理

### 技术栈
- FastAPI: Web框架，提供高性能的API开发支持
- Uvicorn: ASGI服务器，用于运行FastAPI应用
- Python异步编程: 使用async/await处理并发请求
- SSE (Server-Sent Events): 实现流式响应
- tiktoken: OpenAI的分词器，用于token计算
- aiohttp: 异步HTTP客户端库
- colorlog: 彩色日志输出
- Docker: 容器化部署支持
- UV: 现代化的Python包管理器

## 系统要求

### 基础环境
- Python 3.11 或更高版本
- uv 包管理器（推荐）或 pip
- Git（用于版本控制和代码获取）
- Docker（可选，用于容器化部署）

### 硬件要求
- CPU：1核或更高
- 内存：2GB或更高
- 磁盘空间：至少500MB可用空间

### 网络要求
- 稳定的互联网连接
- 能够访问以下API服务：
  - Anthropic API (claude-3系列模型)
  - DeepSeek API (deepseek-reasoner模型)
  - 或者 OpenRouter/OneAPI 等中转服务
- 建议使用HTTPS进行安全通信

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ErlichLiu/DeepClaude&type=Date)](https://star-history.com/#ErlichLiu/DeepClaude&Date)

# Buy me a coffee
<img src="https://img.erlich.fun/personal-blog/uPic/IMG_3625.JPG" alt="微信赞赏码" style="width: 400px;"/>

# About Me
- Email: erlichliu@gmail.com
- Website: [Erlichliu](https://erlich.fun)