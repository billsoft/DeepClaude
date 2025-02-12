# 项目目录结构
```
.
├── Dockerfile
├── app/
│   ├── main.py
│   ├── clients/
│   │   ├── base_client.py
│   │   ├── __init__.py
│   │   ├── deepseek_client.py
│   │   ├── claude_client.py
│   ├── utils/
│   │   ├── auth.py
│   │   ├── logger.py
│   ├── deepclaude/
│   │   ├── __init__.py
│   │   ├── deepclaude.py
├── .github/
│   ├── workflows/
├── doc/
```

# Web服务器层

## .../app/main.py
```python
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import logger
from app.utils.auth import verify_api_key
from app.deepclaude.deepclaude import DeepClaude
app = FastAPI(title="DeepClaude API")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER", "anthropic")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"
allow_origins_list = ALLOW_ORIGINS.split(",") if ALLOW_ORIGINS else []
app.add_middleware(
 CORSMiddleware,
 allow_origins=allow_origins_list,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)
if not DEEPSEEK_API_KEY or not CLAUDE_API_KEY:
 logger.critical("请设置环境变量 CLAUDE_API_KEY 和 DEEPSEEK_API_KEY")
 sys.exit(1)
deep_claude = DeepClaude(
 DEEPSEEK_API_KEY,
 CLAUDE_API_KEY,
 DEEPSEEK_API_URL,
 CLAUDE_API_URL,
 CLAUDE_PROVIDER,
 IS_ORIGIN_REASONING
)
logger.debug("当前日志级别为 DEBUG")
logger.info("开始请求")
@app.get("/", dependencies=[Depends(verify_api_key)])
async def root():
 logger.info("访问了根路径")
 return {"message": "Welcome to DeepClaude API"}
@app.get("/v1/models")
async def list_models():
 models = [{
 "id": "deepclaude",
 "object": "model",
 "created": 1677610602,
 "owned_by": "deepclaude",
 "permission": [{
 "id": "modelperm-deepclaude",
 "object": "model_permission",
 "created": 1677610602,
 "allow_create_engine": False,
 "allow_sampling": True,
 "allow_logprobs": True,
 "allow_search_indices": False,
 "allow_view": True,
 "allow_fine_tuning": False,
 "organization": "*",
 "group": None,
 "is_blocking": False
 }],
 "root": "deepclaude",
 "parent": None
 }]
 return {"object": "list", "data": models}
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
 try:
 body = await request.json()
 messages = body.get("messages")
 model_arg = (
 get_and_validate_params(body)
 )
 stream = model_arg[4]
 if stream:
 return StreamingResponse(
 deep_claude.chat_completions_with_stream(
 messages=messages,
 model_arg=model_arg[:4],
 deepseek_model=DEEPSEEK_MODEL,
 claude_model=CLAUDE_MODEL
 ),
 media_type="text/event-stream"
 )
 else:
 response = await deep_claude.chat_completions_without_stream(
 messages=messages,
 model_arg=model_arg[:4],
 deepseek_model=DEEPSEEK_MODEL,
 claude_model=CLAUDE_MODEL
 )
 return response
 except Exception as e:
 logger.error(f"处理请求时发生错误: {e}")
 return {"error": str(e)}
def get_and_validate_params(body):
 temperature: float = body.get("temperature", 0.5)
 top_p: float = body.get("top_p", 0.9)
 presence_penalty: float = body.get("presence_penalty", 0.0)
 frequency_penalty: float = body.get("frequency_penalty", 0.0)
 stream: bool = body.get("stream", True)
 if "sonnet" in body.get("model", ""):
 if not isinstance(temperature, (float)) or temperature < 0.0 or temperature > 1.0:
 raise ValueError("Sonnet 设定 temperature 必须在 0 到 1 之间")
 return (temperature, top_p, presence_penalty, frequency_penalty, stream)```
______________________________

## .../clients/base_client.py
```python
from typing import AsyncGenerator, Any
import aiohttp
from app.utils.logger import logger
from abc import ABC, abstractmethod
class BaseClient(ABC):
 def __init__(self, api_key: str, api_url: str):
 self.api_key = api_key
 self.api_url = api_url
 async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
 try:
 async with aiohttp.ClientSession() as session:
 async with session.post(self.api_url, headers=headers, json=data) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"API 请求失败: {error_text}")
 return
 async for chunk in response.content.iter_any():
 yield chunk
 except Exception as e:
 logger.error(f"请求 API 时发生错误: {e}")
 @abstractmethod
 async def stream_chat(self, messages: list, model: str) -> AsyncGenerator[tuple[str, str], None]:
 pass```
______________________________

## .../clients/__init__.py
```python
from .base_client import BaseClient
from .deepseek_client import DeepSeekClient
from .claude_client import ClaudeClient
__all__ = ['BaseClient', 'DeepSeekClient', 'ClaudeClient']```
______________________________

## .../clients/deepseek_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
class DeepSeekClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = "https://api.siliconflow.cn/v1/chat/completions", provider: str = "deepseek"):
 super().__init__(api_key, api_url)
 self.provider = provider
 def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
 has_start = "<think>" in content
 has_end = "</think>" in content
 if has_start and has_end:
 return True, content
 elif has_start:
 return False, content
 elif not has_start and not has_end:
 return False, content
 else:
 return True, content
 async def stream_chat(self, messages: list, model: str = "deepseek-ai/DeepSeek-R1", is_origin_reasoning: bool = True) -> AsyncGenerator[tuple[str, str], None]:
 headers = {
 "Authorization": f"Bearer {self.api_key}",
 "Content-Type": "application/json",
 "Accept": "text/event-stream",
 }
 data = {
 "model": model,
 "messages": messages,
 "stream": True,
 }
 logger.debug(f"开始流式对话：{data}")
 accumulated_content = ""
 is_collecting_think = False
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 try:
 lines = chunk_str.splitlines()
 for line in lines:
 if line.startswith("data: "):
 json_str = line[len("data: "):]
 if json_str == "[DONE]":
 return
 data = json.loads(json_str)
 if data and data.get("choices") and data["choices"][0].get("delta"):
 delta = data["choices"][0]["delta"]
 if is_origin_reasoning:
 if delta.get("reasoning_content"):
 content = delta["reasoning_content"]
 logger.debug(f"提取推理内容：{content}")
 yield "reasoning", content
 if delta.get("reasoning_content") is None and delta.get("content"):
 content = delta["content"]
 logger.info(f"提取内容信息，推理阶段结束: {content}")
 yield "content", content
 else:
 if delta.get("content"):
 content = delta["content"]
 if content == "":
 continue
 logger.debug(f"非原生推理内容：{content}")
 accumulated_content += content
 is_complete, processed_content = self._process_think_tag_content(accumulated_content)
 if "<think>" in content and not is_collecting_think:
 logger.debug(f"开始收集推理内容：{content}")
 is_collecting_think = True
 yield "reasoning", content
 elif is_collecting_think:
 if "</think>" in content:
 logger.debug(f"推理内容结束：{content}")
 is_collecting_think = False
 yield "reasoning", content
 yield "content", ""
 accumulated_content = ""
 else:
 yield "reasoning", content
 else:
 yield "content", content
 except json.JSONDecodeError as e:
 logger.error(f"JSON 解析错误: {e}")
 except Exception as e:
 logger.error(f"处理 chunk 时发生错误: {e}")```
______________________________

## .../clients/claude_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
class ClaudeClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages", provider: str = "anthropic"):
 super().__init__(api_key, api_url)
 self.provider = provider
 async def stream_chat(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 model: str,
 stream: bool = True
 ) -> AsyncGenerator[tuple[str, str], None]:
 if self.provider == "openrouter":
 model = "anthropic/claude-3.5-sonnet"
 headers = {
 "Authorization": f"Bearer {self.api_key}",
 "Content-Type": "application/json",
 "HTTP-Referer": "https://github.com/ErlichLiu/DeepClaude",
 "X-Title": "DeepClaude"
 }
 data = {
 "model": model,
 "messages": messages,
 "stream": stream,
 "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
 "top_p": model_arg[1],
 "presence_penalty": model_arg[2],
 "frequency_penalty": model_arg[3]
 }
 elif self.provider == "oneapi":
 headers = {
 "Authorization": f"Bearer {self.api_key}",
 "Content-Type": "application/json"
 }
 data = {
 "model": model,
 "messages": messages,
 "stream": stream,
 "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
 "top_p": model_arg[1],
 "presence_penalty": model_arg[2],
 "frequency_penalty": model_arg[3]
 }
 elif self.provider == "anthropic":
 headers = {
 "x-api-key": self.api_key,
 "anthropic-version": "2023-06-01",
 "content-type": "application/json",
 "accept": "text/event-stream" if stream else "application/json",
 }
 data = {
 "model": model,
 "messages": messages,
 "max_tokens": 8192,
 "stream": stream,
 "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
 "top_p": model_arg[1]
 }
 else:
 raise ValueError(f"不支持的Claude Provider: {self.provider}")
 logger.debug(f"开始对话：{data}")
 if stream:
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 for line in chunk_str.split('\n'):
 if line.startswith('data: '):
 json_str = line[6:]
 if json_str.strip() == '[DONE]':
 return
 try:
 data = json.loads(json_str)
 if self.provider in ("openrouter", "oneapi"):
 content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
 if content:
 yield "answer", content
 elif self.provider == "anthropic":
 if data.get('type') == 'content_block_delta':
 content = data.get('delta', {}).get('text', '')
 if content:
 yield "answer", content
 else:
 raise ValueError(f"不支持的Claude Provider: {self.provider}")
 except json.JSONDecodeError:
 continue
 else:
 async for chunk in self._make_request(headers, data):
 try:
 response = json.loads(chunk.decode('utf-8'))
 if self.provider in ("openrouter", "oneapi"):
 content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
 if content:
 yield "answer", content
 elif self.provider == "anthropic":
 content = response.get('content', [{}])[0].get('text', '')
 if content:
 yield "answer", content
 else:
 raise ValueError(f"不支持的Claude Provider: {self.provider}")
 except json.JSONDecodeError:
 continue```
______________________________

## .../utils/auth.py
```python
from fastapi import HTTPException, Header
from typing import Optional
import os
from dotenv import load_dotenv
from app.utils.logger import logger
logger.info(f"当前工作目录: {os.getcwd()}")
logger.info("尝试加载.env文件...")
load_dotenv(override=True)
ALLOW_API_KEY = os.getenv("ALLOW_API_KEY")
logger.info(f"ALLOW_API_KEY环境变量状态: {'已设置' if ALLOW_API_KEY else '未设置'}")
if not ALLOW_API_KEY:
 raise ValueError("ALLOW_API_KEY environment variable is not set")
logger.info(f"Loaded API key starting with: {ALLOW_API_KEY[:4] if len(ALLOW_API_KEY) >= 4 else ALLOW_API_KEY}")
async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
 if authorization is None:
 logger.warning("请求缺少Authorization header")
 raise HTTPException(
 status_code=401,
 detail="Missing Authorization header"
 )
 api_key = authorization.replace("Bearer ", "").strip()
 if api_key != ALLOW_API_KEY:
 logger.warning(f"无效的API密钥: {api_key}")
 raise HTTPException(
 status_code=401,
 detail="Invalid API key"
 )
 logger.info("API密钥验证通过")```
______________________________

## .../utils/logger.py
```python
import logging
import colorlog
import sys
import os
from dotenv import load_dotenv
load_dotenv()
def get_log_level() -> int:
 level_map = {
 'DEBUG': logging.DEBUG,
 'INFO': logging.INFO,
 'WARNING': logging.WARNING,
 'ERROR': logging.ERROR,
 'CRITICAL': logging.CRITICAL
 }
 level = os.getenv('LOG_LEVEL', 'INFO').upper()
 return level_map.get(level, logging.INFO)
def setup_logger(name: str = "DeepClaude") -> logging.Logger:
 logger = colorlog.getLogger(name)
 if logger.handlers:
 return logger
 log_level = get_log_level()
 logger.setLevel(log_level)
 console_handler = logging.StreamHandler(sys.stdout)
 console_handler.setLevel(log_level)
 formatter = colorlog.ColoredFormatter(
 "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
 datefmt="%Y-%m-%d %H:%M:%S",
 log_colors={
 'DEBUG': 'cyan',
 'INFO': 'green',
 'WARNING': 'yellow',
 'ERROR': 'red',
 'CRITICAL': 'red,bg_white',
 }
 )
 console_handler.setFormatter(formatter)
 logger.addHandler(console_handler)
 return logger
logger = setup_logger()```
______________________________

## .../deepclaude/__init__.py
```python
```
______________________________

## .../deepclaude/deepclaude.py
```python
import json
import time
import tiktoken
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient
class DeepClaude:
 def __init__(self, deepseek_api_key: str, claude_api_key: str,
 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
 claude_api_url: str = "https://api.anthropic.com/v1/messages",
 claude_provider: str = "anthropic",
 is_origin_reasoning: bool = True):
 self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
 self.claude_client = ClaudeClient(claude_api_key, claude_api_url, claude_provider)
 self.is_origin_reasoning = is_origin_reasoning
 async def chat_completions_with_stream(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 deepseek_model: str = "deepseek-reasoner",
 claude_model: str = "claude-3-5-sonnet-20241022"
 ) -> AsyncGenerator[bytes, None]:
 chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
 created_time = int(time.time())
 output_queue = asyncio.Queue()
 claude_queue = asyncio.Queue()
 reasoning_content = []
 async def process_deepseek():
 logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}, 提供商: {self.deepseek_client.provider}")
 try:
 async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model, self.is_origin_reasoning):
 if content_type == "reasoning":
 reasoning_content.append(content)
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": deepseek_model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "reasoning_content": content,
 "content": ""
 }
 }]
 }
 await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
 elif content_type == "content":
 logger.info(f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}")
 await claude_queue.put("".join(reasoning_content))
 break
 except Exception as e:
 logger.error(f"处理 DeepSeek 流时发生错误: {e}")
 await claude_queue.put("")
 logger.info("DeepSeek 任务处理完成，标记结束")
 await output_queue.put(None)
 async def process_claude():
 try:
 logger.info("等待获取 DeepSeek 的推理内容...")
 reasoning = await claude_queue.get()
 logger.debug(f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}")
 if not reasoning:
 logger.warning("未能获取到有效的推理内容，将使用默认提示继续")
 reasoning = "获取推理内容失败"
 claude_messages = messages.copy()
 combined_content = f
 last_message = claude_messages[-1]
 if last_message.get("role", "") == "user":
 original_content = last_message["content"]
 fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
 last_message["content"] = fixed_content
 claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]
 logger.info(f"开始处理 Claude 流，使用模型: {claude_model}, 提供商: {self.claude_client.provider}")
 async for content_type, content in self.claude_client.stream_chat(
 messages=claude_messages,
 model_arg=model_arg,
 model=claude_model,
 ):
 if content_type == "answer":
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": claude_model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": content
 }
 }]
 }
 await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
 except Exception as e:
 logger.error(f"处理 Claude 流时发生错误: {e}")
 logger.info("Claude 任务处理完成，标记结束")
 await output_queue.put(None)
 deepseek_task = asyncio.create_task(process_deepseek())
 claude_task = asyncio.create_task(process_claude())
 finished_tasks = 0
 while finished_tasks < 2:
 item = await output_queue.get()
 if item is None:
 finished_tasks += 1
 else:
 yield item
 yield b'data: [DONE]\n\n'
 async def chat_completions_without_stream(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 deepseek_model: str = "deepseek-reasoner",
 claude_model: str = "claude-3-5-sonnet-20241022"
 ) -> dict:
 chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
 created_time = int(time.time())
 reasoning_content = []
 try:
 async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model, self.is_origin_reasoning):
 if content_type == "reasoning":
 reasoning_content.append(content)
 elif content_type == "content":
 break
 except Exception as e:
 logger.error(f"获取 DeepSeek 推理内容时发生错误: {e}")
 reasoning_content = ["获取推理内容失败"]
 reasoning = "".join(reasoning_content)
 claude_messages = messages.copy()
 combined_content = f
 last_message = claude_messages[-1]
 if last_message.get("role", "") == "user":
 original_content = last_message["content"]
 fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
 last_message["content"] = fixed_content
 claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]
 token_content = "\n".join([message.get("content", "") for message in claude_messages])
 encoding = tiktoken.encoding_for_model("gpt-4o")
 input_tokens = encoding.encode(token_content)
 logger.debug(f"输入 Tokens: {len(input_tokens)}")
 logger.debug("claude messages: " + str(claude_messages))
 try:
 answer = ""
 async for content_type, content in self.claude_client.stream_chat(
 messages=claude_messages,
 model_arg=model_arg,
 model=claude_model,
 stream=False
 ):
 if content_type == "answer":
 answer += content
 output_tokens = encoding.encode(answer)
 logger.debug(f"输出 Tokens: {len(output_tokens)}")
 return {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": claude_model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": answer,
 "reasoning_content": reasoning
 },
 "finish_reason": "stop"
 }],
 "usage": {
 "prompt_tokens": len(input_tokens),
 "completion_tokens": len(output_tokens),
 "total_tokens": len(input_tokens + output_tokens)
 }
 }
 except Exception as e:
 logger.error(f"获取 Claude 响应时发生错误: {e}")
 raise e```
______________________________

# 配置文件

## ./Dockerfile
```dockerfile
# 使用 Python 3.11 slim 版本作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
 PYTHONDONTWRITEBYTECODE=1

# 安装依赖
RUN pip install --no-cache-dir \
 aiohttp==3.11.11 \
 colorlog==6.9.0 \
 fastapi==0.115.8 \
 python-dotenv==1.0.1 \
 tiktoken==0.8.0 \
 "uvicorn[standard]"

# 复制项目文件
COPY ./app ./app

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
______________________________
