# 项目目录结构
```
.
├── Dockerfile
├── main.py
├── .cursor/
│   ├── rules/
├── .github/
│   ├── workflows/
├── app/
│   ├── main.py
│   ├── clients/
│   │   ├── base_client.py
│   │   ├── claude_client.py
│   │   ├── deepseek_client.py
│   │   ├── ollama_r1.py
│   │   ├── __init__.py
│   ├── deepclaude/
│   │   ├── deepclaude.py
│   │   ├── __init__.py
│   ├── utils/
│   │   ├── auth.py
│   │   ├── logger.py
│   │   ├── message_processor.py
├── doc/
├── test/
│   ├── test_claude_client.py
│   ├── test_deepclaude.py
│   ├── test_deepseek_client.py
│   ├── test_nvidia_deepseek.py
│   ├── test_ollama_r1.py
```

# Web服务器层

## main.py
```python
import os
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import logger
load_dotenv()
def main():
 host = os.getenv('HOST', '::')
 port = int(os.getenv('PORT', 1124))
 reload = os.getenv('RELOAD', 'false').lower() == 'true'
 uvicorn.run(
 'app.main:app',
 host=host,
 port=port,
 reload=reload,
 loop='uvloop' if os.name != 'nt' else 'asyncio'
 )
if __name__ == '__main__':
 main()```
______________________________

## .\main.py
```python
import os
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import logger
load_dotenv()
def main():
 host = os.getenv('HOST', '::')
 port = int(os.getenv('PORT', 1124))
 reload = os.getenv('RELOAD', 'false').lower() == 'true'
 uvicorn.run(
 'app.main:app',
 host=host,
 port=port,
 reload=reload,
 loop='uvloop' if os.name != 'nt' else 'asyncio'
 )
if __name__ == '__main__':
 main()```
______________________________

## ...\app\main.py
```python
import os
import sys
from dotenv import load_dotenv
import uuid
import time
load_dotenv()
def setup_proxy():
 enable_proxy = os.getenv('ENABLE_PROXY', 'false').lower() == 'true'
 if enable_proxy:
 os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
 os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
 else:
 os.environ.pop('HTTP_PROXY', None)
 os.environ.pop('HTTPS_PROXY', None)
setup_proxy()
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import logger
from app.utils.auth import verify_api_key
from app.deepclaude.deepclaude import DeepClaude
from fastapi.responses import JSONResponse
app = FastAPI(title="DeepClaude API")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER", "anthropic")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
DEEPSEEK_PROVIDER = os.getenv("DEEPSEEK_PROVIDER", "deepseek")
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "false").lower() == "true"
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "deepseek").lower()
allow_origins_list = ALLOW_ORIGINS.split(",") if ALLOW_ORIGINS else []
app.add_middleware(
 CORSMiddleware,
 allow_origins=allow_origins_list,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
if REASONING_PROVIDER == 'ollama' and not OLLAMA_API_URL:
 logger.critical("使用 Ollama 推理时必须设置 OLLAMA_API_URL")
 sys.exit(1)
if REASONING_PROVIDER == 'deepseek' and not DEEPSEEK_API_KEY:
 logger.critical("使用 DeepSeek 推理时必须设置 DEEPSEEK_API_KEY")
 sys.exit(1)
if not CLAUDE_API_KEY:
 logger.critical("必须设置 CLAUDE_API_KEY")
 sys.exit(1)
deep_claude = DeepClaude(
 claude_api_key=CLAUDE_API_KEY,
 claude_api_url=CLAUDE_API_URL,
 claude_provider=CLAUDE_PROVIDER,
 deepseek_api_key=DEEPSEEK_API_KEY,
 deepseek_api_url=DEEPSEEK_API_URL,
 deepseek_provider=DEEPSEEK_PROVIDER,
 ollama_api_url=OLLAMA_API_URL,
 is_origin_reasoning=IS_ORIGIN_REASONING
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
 data = await request.json()
 if "messages" not in data:
 raise ValueError("Missing messages parameter")
 if data.get("stream", False):
 return StreamingResponse(
 deep_claude.chat_completions_with_stream(
 messages=data["messages"],
 chat_id=f"chatcmpl-{uuid.uuid4()}",
 created_time=int(time.time()),
 model=data.get("model", "deepclaude")
 ),
 media_type="text/event-stream",
 headers={
 "Cache-Control": "no-cache, no-transform",
 "Connection": "keep-alive",
 "Content-Type": "text/event-stream;charset=utf-8",
 "X-Accel-Buffering": "no",
 "Transfer-Encoding": "chunked",
 "Keep-Alive": "timeout=600"
 }
 )
 else:
 response = await deep_claude.chat_completions_without_stream(
 messages=data["messages"],
 model_arg=get_and_validate_params(data)
 )
 return JSONResponse(content=response)
 except ValueError as e:
 logger.warning(f"参数验证错误: {e}")
 return JSONResponse(
 status_code=400,
 content={"error": str(e)}
 )
 except Exception as e:
 logger.error(f"处理请求时发生错误: {e}", exc_info=True)
 return JSONResponse(
 status_code=500,
 content={"error": "Internal server error"}
 )
def get_and_validate_params(body: dict) -> tuple:
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

## ...\clients\base_client.py
```python
from typing import AsyncGenerator, Any, Tuple
import aiohttp
from app.utils.logger import logger
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import asyncio
load_dotenv()
class BaseClient(ABC):
 def __init__(self, api_key: str, api_url: str):
 self.api_key = api_key
 self.api_url = api_url
 def _get_proxy(self) -> str | None:
 use_proxy, proxy = self._get_proxy_config()
 return proxy if use_proxy else None
 @abstractmethod
 def _get_proxy_config(self) -> tuple[bool, str | None]:
 pass
 @abstractmethod
 async def stream_chat(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 pass
 @abstractmethod
 async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 pass
 @abstractmethod
 def _extract_reasoning(self, content: str) -> tuple[bool, str]:
 pass
 async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
 max_retries = 3
 retry_count = 0
 retry_codes = {429, 500, 502, 503, 504}
 while retry_count < max_retries:
 try:
 async with aiohttp.ClientSession() as session:
 async with session.post(
 self.api_url,
 headers=headers,
 json=data,
 proxy=self._get_proxy(),
 timeout=aiohttp.ClientTimeout(
 total=60,
 connect=10,
 sock_read=30
 )
 ) as response:
 if response.status != 200:
 error_msg = await response.text()
 logger.error(f"API请求失败: HTTP {response.status}\n{error_msg}")
 if response.status in retry_codes:
 retry_count += 1
 wait_time = min(2 ** retry_count, 32)
 logger.warning(f"等待 {wait_time} 秒后重试...")
 await asyncio.sleep(wait_time)
 continue
 raise Exception(f"HTTP {response.status}: {error_msg}")
 buffer = bytearray()
 async for chunk in response.content.iter_chunks():
 chunk_data = chunk[0]
 if not chunk_data:
 continue
 buffer.extend(chunk_data)
 while b"\n" in buffer:
 line, remainder = buffer.split(b"\n", 1)
 if line:
 yield line
 buffer = remainder
 if buffer:
 yield bytes(buffer)
 break
 except (aiohttp.ClientError, asyncio.TimeoutError) as e:
 retry_count += 1
 if retry_count >= max_retries:
 logger.error(f"请求重试次数超过上限: {e}")
 raise
 wait_time = min(2 ** retry_count, 32)
 logger.warning(f"网络错误，等待 {wait_time} 秒后重试: {e}")
 await asyncio.sleep(wait_time)```
______________________________

## ...\clients\claude_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
import os
class ClaudeClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages", provider: str = "anthropic"):
 super().__init__(api_key, api_url)
 self.provider = provider.lower()
 def _extract_reasoning(self, content: str) -> tuple[bool, str]:
 return False, ""
 def _get_proxy_config(self) -> tuple[bool, str | None]:
 enable_proxy = os.getenv('CLAUDE_ENABLE_PROXY', 'false').lower() == 'true'
 if enable_proxy:
 http_proxy = os.getenv('HTTP_PROXY')
 https_proxy = os.getenv('HTTPS_PROXY')
 if https_proxy or http_proxy:
 logger.info(f"Claude 客户端使用代理: {https_proxy or http_proxy}")
 else:
 logger.warning("已启用 Claude 代理但未设置代理地址")
 return True, https_proxy or http_proxy
 logger.debug("Claude 客户端未启用代理")
 return False, None
 def _prepare_headers(self) -> dict:
 headers = {
 "Content-Type": "application/json",
 "Accept": "text/event-stream",
 }
 if self.provider == "anthropic":
 headers.update({
 "x-api-key": self.api_key,
 "anthropic-version": "2023-06-01",
 "anthropic-beta": "messages-2023-12-15"
 })
 elif self.provider == "openrouter":
 headers["Authorization"] = f"Bearer {self.api_key}"
 elif self.provider == "oneapi":
 headers["Authorization"] = f"Bearer {self.api_key}"
 return headers
 def _prepare_request_data(self, messages: list, **kwargs) -> dict:
 data = {
 "model": kwargs.get("model", "claude-3-5-sonnet-20241022"),
 "messages": messages,
 "max_tokens": kwargs.get("max_tokens", 8192),
 "temperature": kwargs.get("temperature", 0.7),
 "top_p": kwargs.get("top_p", 0.9),
 "stream": True
 }
 logger.debug(f"Claude请求数据: {messages}")
 return data
 async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[dict, None]:
 try:
 headers = self._prepare_headers()
 data = self._prepare_request_data(messages, **kwargs)
 logger.debug(f"Claude请求数据: {data}")
 async for chunk in self._make_request(headers, data):
 try:
 if chunk:
 text = chunk.decode('utf-8')
 if text.startswith('data: '):
 data = text[6:].strip()
 if data == '[DONE]':
 break
 response = json.loads(data)
 logger.debug(f"Claude响应数据: {response}")
 if 'type' in response:
 if response['type'] == 'content_block_delta':
 content = response['delta'].get('text', '')
 if content:
 yield {"delta": {"content": content}}
 elif 'choices' in response:
 if response['choices'][0].get('delta', {}).get('content'):
 yield response['choices'][0]
 except json.JSONDecodeError as e:
 logger.error(f"解析Claude响应失败: {e}")
 continue
 except Exception as e:
 logger.error(f"Claude流式请求失败: {e}", exc_info=True)
 raise
 async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 return
 yield```
______________________________

## ...\clients\deepseek_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
import os
class DeepSeekClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = None, provider: str = None):
 self.provider = provider or os.getenv('DEEPSEEK_PROVIDER', 'deepseek')
 self.provider_configs = {
 'deepseek': {
 'url': 'https://api.deepseek.com/v1/chat/completions',
 'model': 'deepseek-reasoner'
 },
 'siliconflow': {
 'url': 'https://api.siliconflow.cn/v1/chat/completions',
 'model': 'deepseek-ai/DeepSeek-R1'
 },
 'nvidia': {
 'url': 'https://integrate.api.nvidia.com/v1/chat/completions',
 'model': 'deepseek-ai/deepseek-r1'
 }
 }
 if self.provider not in self.provider_configs:
 raise ValueError(f"不支持的 provider: {self.provider}")
 config = self.provider_configs[self.provider]
 api_url = api_url or os.getenv('DEEPSEEK_API_URL') or config['url']
 super().__init__(api_key, api_url)
 self.default_model = config['model']
 self.is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true'
 def _get_proxy_config(self) -> tuple[bool, str | None]:
 enable_proxy = os.getenv('DEEPSEEK_ENABLE_PROXY', 'false').lower() == 'true'
 if enable_proxy:
 http_proxy = os.getenv('HTTP_PROXY')
 https_proxy = os.getenv('HTTPS_PROXY')
 logger.info(f"DeepSeek 客户端使用代理: {https_proxy or http_proxy}")
 return True, https_proxy or http_proxy
 logger.debug("DeepSeek 客户端未启用代理")
 return False, None
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
 def _extract_reasoning(self, content: str) -> tuple[bool, str]:
 if self.is_origin_reasoning:
 if "reasoning_content" in content:
 return True, content["reasoning_content"]
 return False, ""
 else:
 if "<think>" in content and "</think>" in content:
 start = content.find("<think>") + 7
 end = content.find("</think>")
 if start < end:
 return True, content[start:end].strip()
 return False, ""
 async def stream_chat(self, messages: list, model: str = None, model_arg: tuple = None) -> AsyncGenerator[tuple[str, str], None]:
 if not model:
 model = self.default_model
 if not model:
 raise ValueError("未指定模型且无默认模型")
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
 if self.provider == 'nvidia':
 temperature = model_arg[0] if model_arg else 0.6
 top_p = model_arg[1] if model_arg else 0.7
 data.update({
 "temperature": temperature,
 "top_p": top_p,
 "max_tokens": 4096
 })
 logger.debug(f"开始流式对话：{data}")
 try:
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 try:
 data = json.loads(chunk_str)
 if not data or not data.get("choices") or not data["choices"][0].get("delta"):
 continue
 delta = data["choices"][0]["delta"]
 has_reasoning, reasoning = self._extract_reasoning(delta)
 if has_reasoning:
 yield "reasoning", reasoning
 except json.JSONDecodeError:
 continue
 except Exception as e:
 logger.error(f"流式对话发生错误: {e}", exc_info=True)
 raise
 async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 model_arg = kwargs.get('model_arg')
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
 if self.provider == 'nvidia':
 temperature = model_arg[0] if model_arg else 0.6
 top_p = model_arg[1] if model_arg else 0.7
 data.update({
 "temperature": temperature,
 "top_p": top_p,
 "max_tokens": 4096
 })
 reasoning_buffer = []
 content_buffer = []
 async for chunk in self._make_request(headers, data):
 try:
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 for line in chunk_str.splitlines():
 if line.startswith("data: "):
 json_str = line[len("data: "):]
 if json_str == "[DONE]":
 continue
 data = json.loads(json_str)
 if not data or not data.get("choices") or not data["choices"][0].get("delta"):
 continue
 delta = data["choices"][0]["delta"]
 content = delta.get("content", "")
 if self.is_origin_reasoning:
 if "reasoning_content" in delta:
 yield "reasoning", delta["reasoning_content"]
 else:
 if "<think>" in content:
 start = content.find("<think>") + len("<think>")
 end = content.find("</think>")
 if end > start:
 reasoning = content[start:end].strip()
 if reasoning:
 yield "reasoning", reasoning
 elif content:
 yield "content", content
 except json.JSONDecodeError:
 continue
 except Exception as e:
 logger.error(f"处理推理内容时发生错误: {e}")
 continue```
______________________________

## ...\clients\ollama_r1.py
```python
import os
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
import asyncio
class OllamaR1Client(BaseClient):
 def __init__(self, api_url: str = "http://localhost:11434"):
 if not api_url:
 raise ValueError("必须提供 Ollama API URL")
 if not api_url.endswith("/api/chat"):
 api_url = f"{api_url.rstrip('/')}/api/chat"
 super().__init__(api_key="", api_url=api_url)
 self.default_model = "deepseek-r1:32b"
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
 def _extract_reasoning(self, content: str) -> tuple[bool, str]:
 if "<think>" in content and "</think>" in content:
 start = content.find("<think>") + 7
 end = content.find("</think>")
 if start < end:
 return True, content[start:end].strip()
 return False, ""
 async def stream_chat(self, messages: list, model: str = "deepseek-r1:32b") -> AsyncGenerator[tuple[str, str], None]:
 if not messages:
 raise ValueError("消息列表不能为空")
 headers = {
 "Content-Type": "application/json",
 }
 data = {
 "model": model,
 "messages": messages,
 "stream": True,
 "options": {
 "temperature": 0.7,
 "num_predict": 1024,
 }
 }
 logger.debug(f"开始流式对话：{data}")
 try:
 current_content = ""
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 try:
 response = json.loads(chunk_str)
 if "message" in response and "content" in response["message"]:
 content = response["message"]["content"]
 has_think_start = "<think>" in content
 has_think_end = "</think>" in content
 if has_think_start and not has_think_end:
 current_content = content
 elif has_think_end and current_content:
 full_content = current_content + content
 has_reasoning, reasoning = self._extract_reasoning(full_content)
 if has_reasoning:
 yield "reasoning", reasoning
 current_content = ""
 elif current_content:
 current_content += content
 else:
 yield "content", content
 if response.get("done"):
 if current_content:
 has_reasoning, reasoning = self._extract_reasoning(current_content)
 if has_reasoning:
 yield "reasoning", reasoning
 return
 except json.JSONDecodeError:
 continue
 except Exception as e:
 logger.error(f"流式对话发生错误: {e}", exc_info=True)
 raise
 async def get_reasoning(self, messages: list, model: str = None, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 if model is None:
 model = self.default_model
 async for content_type, content in self.stream_chat(
 messages=messages,
 model=model
 ):
 if content_type == "reasoning":
 yield content_type, content
 def _get_proxy_config(self) -> tuple[bool, str | None]:
 enable_proxy = os.getenv('OLLAMA_ENABLE_PROXY', 'false').lower() == 'true'
 if enable_proxy:
 http_proxy = os.getenv('HTTP_PROXY')
 https_proxy = os.getenv('HTTPS_PROXY')
 if https_proxy or http_proxy:
 logger.info(f"Ollama 客户端使用代理: {https_proxy or http_proxy}")
 else:
 logger.warning("已启用 Ollama 代理但未设置代理地址")
 return True, https_proxy or http_proxy
 logger.debug("Ollama 客户端未启用代理")
 return False, None```
______________________________

## ...\clients\__init__.py
```python
from .base_client import BaseClient
from .deepseek_client import DeepSeekClient
from .claude_client import ClaudeClient
from .ollama_r1 import OllamaR1Client
__all__ = ['BaseClient', 'DeepSeekClient', 'ClaudeClient', 'OllamaR1Client']```
______________________________

## ...\deepclaude\deepclaude.py
```python
import json
import time
import tiktoken
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, OllamaR1Client
from app.utils.message_processor import MessageProcessor
import aiohttp
import os
from dotenv import load_dotenv
load_dotenv()
class DeepClaude:
 def __init__(self, **kwargs):
 self.claude_client = ClaudeClient(
 api_key=kwargs.get('claude_api_key'),
 api_url=kwargs.get('claude_api_url'),
 provider=kwargs.get('claude_provider')
 )
 self.reasoning_providers = {
 'deepseek': lambda: DeepSeekClient(
 api_key=kwargs.get('deepseek_api_key'),
 api_url=kwargs.get('deepseek_api_url'),
 provider=kwargs.get('deepseek_provider')
 ),
 'ollama': lambda: OllamaR1Client(
 api_url=kwargs.get('ollama_api_url')
 )
 }
 def _get_reasoning_provider(self):
 provider = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 if provider not in self.reasoning_providers:
 raise ValueError(f"不支持的推理提供者: {provider}")
 return self.reasoning_providers[provider]()
 async def _handle_stream_response(self, response_queue: asyncio.Queue,
 chat_id: str, created_time: int, model: str) -> AsyncGenerator[bytes, None]:
 try:
 while True:
 item = await response_queue.get()
 if item is None:
 break
 content_type, content = item
 if content_type == "reasoning":
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"🤔 思考过程:\n{content}\n"
 }
 }]
 }
 elif content_type == "content":
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"\n\n---\n思考完毕，开始回答：\n\n{content}"
 }
 }]
 }
 yield f"data: {json.dumps(response)}\n\n".encode('utf-8')
 except Exception as e:
 logger.error(f"处理流式响应时发生错误: {e}", exc_info=True)
 raise
 async def _handle_api_error(self, e: Exception) -> str:
 if isinstance(e, aiohttp.ClientError):
 return "网络连接错误，请检查网络连接"
 elif isinstance(e, asyncio.TimeoutError):
 return "请求超时，请稍后重试"
 elif isinstance(e, ValueError):
 return f"参数错误: {str(e)}"
 else:
 return f"未知错误: {str(e)}"
 async def chat_completions_with_stream(self, messages: list, **kwargs):
 try:
 logger.info("开始流式处理请求...")
 provider = self._get_reasoning_provider()
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 **self._prepare_thinker_kwargs(kwargs)
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 yield self._format_stream_response(content, **kwargs)
 if not reasoning_content:
 logger.warning("未获取到思考内容，使用原始问题")
 reasoning_content = [messages[-1]["content"]]
 prompt = self._format_claude_prompt(
 messages[-1]['content'],
 "\n".join(reasoning_content)
 )
 async for chunk in self.claude_client.stream_chat(
 messages=[{"role": "user", "content": prompt}],
 **self._prepare_answerer_kwargs(kwargs)
 ):
 if chunk and "content" in chunk.get("delta", {}):
 yield self._format_stream_response(
 chunk["delta"]["content"],
 **kwargs
 )
 except Exception as e:
 error_msg = await self._handle_api_error(e)
 logger.error(f"流式处理错误: {error_msg}", exc_info=True)
 yield self._format_stream_response(f"错误: {error_msg}", **kwargs)
 def _prepare_thinker_kwargs(self, kwargs: dict) -> dict:
 provider_type = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 if provider_type == 'ollama':
 model = "deepseek-r1:32b"
 else:
 model = kwargs.get('model', 'deepseek-ai/DeepSeek-R1')
 return {
 'model': model,
 'temperature': kwargs.get('temperature', 0.7),
 'top_p': kwargs.get('top_p', 0.9)
 }
 def _prepare_answerer_kwargs(self, kwargs: dict) -> dict:
 return {
 'model': 'claude-3-5-sonnet-20241022',
 'temperature': kwargs.get('temperature', 0.7),
 'top_p': kwargs.get('top_p', 0.9)
 }
 def _chunk_content(self, content: str, chunk_size: int = 3) -> list[str]:
 return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
 def _format_claude_prompt(self, original_question: str, reasoning: str) -> str:
 return f
 async def chat_completions_without_stream(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 deepseek_model: str = "deepseek-reasoner",
 claude_model: str = "claude-3-5-sonnet-20241022"
 ) -> dict:
 logger.info("开始处理请求...")
 logger.debug(f"输入消息: {messages}")
 logger.info("正在获取推理内容...")
 try:
 reasoning = await self._get_reasoning_content(
 messages=messages,
 model=deepseek_model,
 model_arg=model_arg
 )
 except Exception as e:
 logger.error(f"获取推理内容失败: {e}")
 reasoning = "无法获取推理内容"
 logger.debug(f"获取到推理内容: {reasoning}")
 combined_content = f
 claude_messages = [{"role": "user", "content": combined_content}]
 logger.info("正在获取 Claude 回答...")
 try:
 async for content_type, content in self.claude_client.stream_chat(
 messages=claude_messages,
 model_arg=model_arg,
 model=claude_model,
 stream=False
 ):
 if content_type == "answer":
 logger.debug(f"获取到 Claude 回答: {content}")
 return {
 "content": content,
 "role": "assistant"
 }
 except Exception as e:
 logger.error(f"获取 Claude 回答失败: {e}")
 raise
 async def _get_reasoning_content(self, messages: list, model: str, **kwargs) -> str:
 try:
 provider = self._get_reasoning_provider()
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model=model,
 model_arg=kwargs.get('model_arg')
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 return "\n".join(reasoning_content)
 except Exception as e:
 logger.error(f"主要推理提供者失败: {e}")
 if hasattr(self, 'ollama_api_url'):
 logger.info("尝试切换到 Ollama 推理提供者")
 try:
 provider = OllamaR1Client(self.ollama_api_url)
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model="deepseek-r1:32b"
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 return "\n".join(reasoning_content)
 except Exception as e:
 logger.error(f"备用推理提供者也失败: {e}")
 return "无法获取推理内容"
 async def _retry_operation(self, operation, max_retries=3):
 for i in range(max_retries):
 try:
 return await operation()
 except Exception as e:
 if i == max_retries - 1:
 raise
 logger.warning(f"操作失败，正在重试 ({i+1}/{max_retries}): {str(e)}")
 await asyncio.sleep(1 * (i + 1))
 def _validate_model_names(self, deepseek_model: str, claude_model: str):
 if not deepseek_model or not isinstance(deepseek_model, str):
 raise ValueError("无效的 DeepSeek 模型名称")
 if not claude_model or not isinstance(claude_model, str):
 raise ValueError("无效的 Claude 模型名称")
 def _validate_messages(self, messages: list) -> None:
 if not messages:
 raise ValueError("消息列表不能为空")
 for msg in messages:
 if not isinstance(msg, dict):
 raise ValueError("消息必须是字典格式")
 if "role" not in msg or "content" not in msg:
 raise ValueError("消息必须包含 role 和 content 字段")
 if msg["role"] not in ["user", "assistant", "system"]:
 raise ValueError(f"不支持的消息角色: {msg['role']}")
 async def _get_reasoning_with_fallback(
 self,
 messages: list,
 model: str,
 model_arg: tuple[float, float, float, float] = None
 ) -> str:
 try:
 provider = self._get_reasoning_provider()
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model=model,
 model_arg=model_arg
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 return "".join(reasoning_content)
 except Exception as e:
 logger.error(f"主要推理提供者失败: {e}")
 if isinstance(provider, DeepSeekClient):
 logger.info("尝试切换到 Ollama 推理提供者")
 provider = OllamaR1Client(self.ollama_api_url)
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model="deepseek-r1:32b"
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 return "".join(reasoning_content)
 raise
 def _validate_config(self):
 provider = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 if provider == 'deepseek':
 if not self.deepseek_api_key:
 raise ValueError("使用 DeepSeek 时必须提供 API KEY")
 if not self.deepseek_api_url:
 raise ValueError("使用 DeepSeek 时必须提供 API URL")
 elif provider == 'ollama':
 if not self.ollama_api_url:
 raise ValueError("使用 Ollama 时必须提供 API URL")
 def _format_stream_response(self, content: str, **kwargs) -> bytes:
 response = {
 "id": kwargs.get("chat_id", f"chatcmpl-{int(time.time())}"),
 "object": "chat.completion.chunk",
 "created": kwargs.get("created_time", int(time.time())),
 "model": kwargs.get("model", "deepclaude"),
 "choices": [{
 "index": 0,
 "delta": {
 "content": content
 }
 }]
 }
 return f"data: {json.dumps(response)}\n\n".encode('utf-8')
 def _validate_kwargs(self, kwargs: dict) -> None:
 temperature = kwargs.get('temperature')
 if temperature is not None:
 if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1:
 raise ValueError("temperature 必须在 0 到 1 之间")
 top_p = kwargs.get('top_p')
 if top_p is not None:
 if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
 raise ValueError("top_p 必须在 0 到 1 之间")
 model = kwargs.get('model')
 if model and not isinstance(model, str):
 raise ValueError("model 必须是字符串类型")
 def _split_into_tokens(self, text: str) -> list[str]:
 return list(text)```
______________________________

## ...\deepclaude\__init__.py
```python
```
______________________________

## ...\utils\auth.py
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

## ...\utils\logger.py
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

## ...\utils\message_processor.py
```python
from typing import List, Dict
from app.utils.logger import logger
class MessageProcessor:
 @staticmethod
 def convert_to_deepseek_format(messages: List[Dict]) -> List[Dict]:
 processed = []
 temp_content = []
 current_role = None
 for msg in messages:
 role = msg.get("role", "")
 content = msg.get("content", "")
 if not content:
 continue
 if role == "system":
 if processed and processed[0]["role"] == "system":
 processed[0]["content"] += f"\n{content}"
 else:
 processed.insert(0, {"role": "system", "content": content})
 continue
 if role == current_role:
 temp_content.append(content)
 else:
 if temp_content:
 processed.append({
 "role": current_role,
 "content": "\n".join(temp_content)
 })
 temp_content = [content]
 current_role = role
 if temp_content:
 processed.append({
 "role": current_role,
 "content": "\n".join(temp_content)
 })
 final_messages = []
 for i, msg in enumerate(processed):
 if i > 0 and msg["role"] == final_messages[-1]["role"]:
 if msg["role"] == "user":
 final_messages.append({"role": "assistant", "content": "请继续。"})
 else:
 final_messages.append({"role": "user", "content": "请继续。"})
 final_messages.append(msg)
 logger.debug(f"转换后的消息格式: {final_messages}")
 return final_messages
 @staticmethod
 def validate_messages(messages: List[Dict]) -> bool:
 if not messages:
 return False
 for i in range(1, len(messages)):
 if messages[i]["role"] == messages[i-1]["role"]:
 return False
 return True```
______________________________

## ...\test\test_claude_client.py
```python
import os
import sys
import asyncio
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.claude_client import ClaudeClient
from app.utils.logger import logger
load_dotenv()
async def test_claude_stream():
 api_key = os.getenv("CLAUDE_API_KEY")
 api_url = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
 provider = os.getenv("CLAUDE_PROVIDER", "anthropic")
 logger.info(f"API URL: {api_url}")
 logger.info(f"API Key 是否存在: {bool(api_key)}")
 logger.info(f"Provider: {provider}")
 enable_proxy = os.getenv('CLAUDE_ENABLE_PROXY', 'false').lower() == 'true'
 if enable_proxy:
 proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
 logger.info(f"代理已启用: {proxy}")
 else:
 logger.info("代理未启用")
 if not api_key:
 logger.error("请在 .env 文件中设置 CLAUDE_API_KEY")
 return
 messages = [
 {"role": "user", "content": "陵水好玩嘛?"}
 ]
 client = ClaudeClient(api_key, api_url, provider)
 try:
 logger.info("开始测试 Claude 流式输出...")
 async for content_type, content in client.stream_chat(
 messages=messages,
 model_arg=(0.7, 0.9, 0, 0),
 model="claude-3-5-sonnet-20241022"
 ):
 if content_type == "answer":
 logger.info(f"收到回答内容: {content}")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
def main():
 asyncio.run(test_claude_stream())
if __name__ == "__main__":
 main()```
______________________________

## ...\test\test_deepclaude.py
```python
import os
async def test_reasoning_fallback():
 deepclaude = DeepClaude(...)
 messages = [{"role": "user", "content": "测试问题"}]
 os.environ['REASONING_PROVIDER'] = 'deepseek'
 reasoning = await deepclaude._get_reasoning_with_fallback(
 messages=messages,
 model="deepseek-reasoner"
 )
 assert reasoning```
______________________________

## ...\test\test_deepseek_client.py
```python
import os
import sys
import asyncio
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger
load_dotenv()
os.environ['LOG_LEVEL'] = 'DEBUG'
async def test_deepseek_stream():
 api_key = os.getenv("DEEPSEEK_API_KEY")
 api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
 is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"
 logger.info("=== DeepSeek 客户端测试开始 ===")
 logger.info(f"API URL: {api_url}")
 logger.info(f"API Key 是否存在: {bool(api_key)}")
 logger.info(f"原始推理模式: {is_origin_reasoning}")
 if not api_key:
 logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
 return
 messages = [
 {"role": "user", "content": "1+1等于几?"}
 ]
 client = DeepSeekClient(api_key, api_url)
 try:
 logger.info("开始测试 DeepSeek 流式输出...")
 logger.debug(f"发送消息: {messages}")
 reasoning_buffer = []
 content_buffer = []
 async for content_type, content in client.stream_chat(
 messages=messages,
 model="deepseek-ai/DeepSeek-R1",
 is_origin_reasoning=is_origin_reasoning
 ):
 if content_type == "reasoning":
 reasoning_buffer.append(content)
 if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
 reasoning_buffer = []
 elif content_type == "content":
 content_buffer.append(content)
 if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"最终答案：{''.join(content_buffer)}")
 content_buffer = []
 if reasoning_buffer:
 logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
 if content_buffer:
 logger.info(f"最终答案：{''.join(content_buffer)}")
 logger.info("=== DeepSeek 客户端测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
def main():
 asyncio.run(test_deepseek_stream())
if __name__ == "__main__":
 main()```
______________________________

## ...\test\test_nvidia_deepseek.py
```python
import os
import sys
import asyncio
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger
load_dotenv()
async def test_nvidia_deepseek_stream():
 api_key = os.getenv("DEEPSEEK_API_KEY")
 api_url = os.getenv("DEEPSEEK_API_URL")
 logger.info("=== NVIDIA DeepSeek 客户端测试开始 ===")
 logger.info(f"API URL: {api_url}")
 logger.info(f"API Key 是否存在: {bool(api_key)}")
 if not api_key:
 logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
 return
 messages = [
 {"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}
 ]
 client = DeepSeekClient(
 api_key=api_key,
 api_url=api_url,
 provider="nvidia"
 )
 try:
 logger.info("开始测试 NVIDIA DeepSeek 流式输出...")
 logger.debug(f"发送消息: {messages}")
 reasoning_buffer = []
 content_buffer = []
 async for content_type, content in client.stream_chat(
 messages=messages,
 model="deepseek-ai/deepseek-r1"
 ):
 if content_type == "reasoning":
 reasoning_buffer.append(content)
 if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
 reasoning_buffer = []
 elif content_type == "content":
 content_buffer.append(content)
 if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"最终答案：{''.join(content_buffer)}")
 content_buffer = []
 if reasoning_buffer:
 logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
 if content_buffer:
 logger.info(f"最终答案：{''.join(content_buffer)}")
 logger.info("=== NVIDIA DeepSeek 客户端测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
def main():
 asyncio.run(test_nvidia_deepseek_stream())
if __name__ == "__main__":
 main()```
______________________________

## ...\test\test_ollama_r1.py
```python
import os
import sys
import asyncio
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.ollama_r1 import OllamaR1Client
from app.utils.logger import logger
load_dotenv()
os.environ['LOG_LEVEL'] = 'DEBUG'
async def test_ollama_stream():
 api_url = os.getenv("OLLAMA_API_URL", "http://192.168.100.81:11434/api/chat")
 logger.info(f"API URL: {api_url}")
 client = OllamaR1Client(api_url)
 try:
 messages = [
 {"role": "user", "content": "9.9和9.11谁大?"}
 ]
 logger.info("开始测试 Ollama R1 流式输出...")
 logger.debug(f"发送消息: {messages}")
 async for msg_type, content in client.stream_chat(messages):
 if msg_type == "reasoning":
 logger.info(f"推理过程: {content}")
 else:
 logger.info(f"最终答案: {content}")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {e}", exc_info=True)
 raise
async def test_ollama_connection():
 api_url = os.getenv("OLLAMA_API_URL")
 assert api_url, "OLLAMA_API_URL 未设置"
 client = OllamaR1Client(api_url)
 messages = [{"role": "user", "content": "测试连接"}]
 try:
 async for _, _ in client.stream_chat(messages):
 pass
 return True
 except Exception as e:
 logger.error(f"Ollama 连接测试失败: {e}")
 return False
if __name__ == "__main__":
 asyncio.run(test_ollama_stream())```
______________________________

# 配置文件

## .\Dockerfile
```dockerfile
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
```
______________________________
