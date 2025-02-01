<div>
<h1>DeepClaude 🐬🧠 - OpenAI Compatible</h1>

<a href="https://github.com/getasterisk/deepclaude"> Inspiration from getasterisk/deepclaude</a>

[![GitHub license](https://img.erlich.fun/personal-blog/uPic/deepclaude.svg)](#)
[![Compatible with](https://img.shields.io/badge/-ChatGPT-412991?style=flat-square&logo=openai&logoColor=FFFFFF)](https://openai.com)

</div>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Implementation](#implementation)
- [How to run](#how-to-run)
  - [1. 获得运行所需的 API](#1-获得运行所需的-api)
  - [2. 开始运行](#2-开始运行)
- [Deployment](#deployment)
- [Technology Stack](#technology-stack)
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

1. 获取 DeepSeek API，因为最近 DeepSeek 还在遭受攻击，所以经常无法使用，推荐使用 Siliconflow 的效果更好（也可以本地 Ollama 的）: https://cloud.siliconflow.cn/i/RXikvHE2
2. 获取 Claude 的 API KEY （目前还没有做中转模式，以及对 Google 和 AWS 托管的版本的兼容支持，欢迎 PR）：https://console.anthropic.com
   注：`但是！大家也可以联系我，我可以为大家提供按量计费的 DeepClaude 的直接 API 服务！微信：geekthings`

## 2. 开始运行
Step 1. 克隆本项目到适合的文件夹并进入项目

```bash
git clone git@github.com:ErlichLiu/DeepClaude.git
cd DeepClaude
```

Step 2. 通过 uv 安装依赖（如果你还没有安装 uv，请看下方注解）

```bash
# 通过 uv 在本地创建虚拟环境，并安装依赖
uv sync
# macOS 激活虚拟环境
source .venv/bin/activate
# Windows 激活虚拟环境
.venv\Scripts\activate
```

Step 3. 配置环境变量

```bash
# 复制 .env 环境变量到本地
cp .env.example .env
```

Step 4. 按照环境变量当中的注释依次填写配置信息（在此步骤可以配置 Ollama）

Step 5. 本地运行程序

```bash
# 本地运行
uvicorn app.main:app
```

Step 6. 配置程序到你的 Chatbox（推荐 [NextChat](https://nextchat.dev/)、[ChatBox](https://chatboxai.app/zh)、[LobeChat](https://lobechat.com/)）

```bash
# 通常 baseUrl 为：http://127.0.0.1:8000/v1
```

**注：本项目采用 uv 作为包管理器，这是一个更快速更现代的管理方式，用于替代 pip，你可以[在此了解更多](https://docs.astral.sh/uv/)****



# Deployment

> 项目支持 Docker 服务器部署，可自行调用接入常用的 Chatbox，也可以作为渠道一直，将其视为一个特殊的 `DeepClaude`模型接入到 `OneAPI` 等产品使用。

以 Dokploy 部署为例，也推荐 Zeabur 和 Railway。

更新中... 预计一天内完成更新（肩膀真的太酸了）

# Technology Stack
- [FastAPI](https://fastapi.tiangolo.com/)
- [UV as package manager](https://docs.astral.sh/uv/#project-management)
- [Docker](https://www.docker.com/)

# Buy me a coffee
<img src="https://img.erlich.fun/personal-blog/uPic/IMG_3625.JPG" alt="微信赞赏码" style="width: 200px;"/>

# About Me
- Email: erlichliu@gmail.com
- Website: [Erlichliu](https://erlich.fun)
  