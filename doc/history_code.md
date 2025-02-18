# 项目目录结构
```
.
├── Dockerfile
├── .github/
│   ├── workflows/
├── app/
│   ├── main.py
│   ├── clients/
│   │   ├── base_client.py
│   │   ├── claude_client.py
│   │   ├── deepseek_client.py
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
│   ├── test_deepseek_client.py
```

# Web服务器层

## ...\app\main.py
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
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
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
 logger.debug(f"收到请求数据: {body}")
 messages = body.get("messages")
 logger.debug(f"消息内容: {messages}")
 processed_messages = []
 for i, msg in enumerate(messages):
 if i == 0 or msg.get("role") != processed_messages[-1].get("role"):
 processed_messages.append(msg)
 else:
 processed_messages[-1]["content"] += f"\n{msg.get('content', '')}"
 model_arg = get_and_validate_params(body)
 stream = model_arg[4]
 if stream:
 try:
 logger.debug(f"开始流式处理，使用处理后的消息: {processed_messages}")
 stream_response = deep_claude.chat_completions_with_stream(
 messages=processed_messages,
 model_arg=model_arg[:4],
 deepseek_model=DEEPSEEK_MODEL,
 claude_model=CLAUDE_MODEL
 )
 return StreamingResponse(
 stream_response,
 media_type="text/event-stream",
 headers={
 "X-Accel-Buffering": "no",
 "Cache-Control": "no-cache",
 "Connection": "keep-alive",
 }
 )
 except ValueError as e:
 error_msg = str(e)
 logger.warning(f"业务逻辑错误: {error_msg}")
 return {"error": True, "message": error_msg}
 except Exception as e:
 error_msg = f"流式处理错误: {str(e)}"
 logger.error(error_msg, exc_info=True)
 return {"error": True, "message": "network error"}
 else:
 try:
 response = await deep_claude.chat_completions_without_stream(
 messages=processed_messages,
 model_arg=model_arg[:4],
 deepseek_model=DEEPSEEK_MODEL,
 claude_model=CLAUDE_MODEL
 )
 return response
 except ValueError as e:
 error_msg = str(e)
 logger.warning(f"业务逻辑错误: {error_msg}")
 return {"error": True, "message": error_msg}
 except Exception as e:
 error_msg = f"非流式处理错误: {str(e)}"
 logger.error(error_msg, exc_info=True)
 return {"error": True, "message": "network error"}
 except Exception as e:
 error_msg = f"处理请求时发生错误: {str(e)}"
 logger.error(error_msg, exc_info=True)
 return {"error": True, "message": "network error"}
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
 logger.debug(f"正在发送请求到: {self.api_url}")
 logger.debug(f"请求头: {headers}")
 logger.debug(f"请求数据: {data}")
 async with session.post(self.api_url, headers=headers, json=data) as response:
 if response.status != 200:
 error_text = await response.text()
 error_msg = (
 f"API 请求失败:\n"
 f"状态码: {response.status}\n"
 f"URL: {self.api_url}\n"
 f"错误信息: {error_text}"
 )
 logger.error(error_msg)
 raise aiohttp.ClientError(error_msg)
 async for chunk in response.content.iter_any():
 if not chunk:
 logger.warning("收到空响应块")
 continue
 yield chunk
 except aiohttp.ClientError as e:
 error_msg = f"网络请求错误: {str(e)}"
 logger.error(error_msg, exc_info=True)
 raise
 except Exception as e:
 error_msg = f"未知错误: {str(e)}"
 logger.error(error_msg, exc_info=True)
 raise
 @abstractmethod
 async def stream_chat(self, messages: list, model: str) -> AsyncGenerator[tuple[str, str], None]:
 pass```
______________________________

## ...\clients\claude_client.py
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

## ...\clients\deepseek_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
VALID_MODELS = ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-Chat-7B"]
class DeepSeekClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = "https://api.siliconflow.cn/v1/chat/completions", provider: str = "deepseek"):
 super().__init__(api_key, api_url)
 self.provider = provider
 self.default_model = "deepseek-ai/DeepSeek-R1"
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
 if model not in VALID_MODELS:
 error_msg = f"无效的模型名称: {model}，可用模型: {VALID_MODELS}"
 logger.error(error_msg)
 raise ValueError(error_msg)
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
 try:
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 for line in chunk_str.splitlines():
 if line.startswith("data: "):
 json_str = line[len("data: "):]
 if json_str == "[DONE]":
 return
 try:
 data = json.loads(json_str)
 if not data or not data.get("choices") or not data["choices"][0].get("delta"):
 continue
 delta = data["choices"][0]["delta"]
 if is_origin_reasoning:
 if delta.get("reasoning_content"):
 content = delta["reasoning_content"]
 logger.debug(f"提取推理内容：{content}")
 yield "reasoning", content
 elif delta.get("content"):
 content = delta["content"]
 logger.info(f"提取内容信息，推理阶段结束: {content}")
 yield "content", content
 else:
 if delta.get("content"):
 content = delta["content"]
 yield "content", content
 except json.JSONDecodeError as e:
 logger.error(f"JSON 解析错误: {e}")
 continue
 except Exception as e:
 logger.error(f"流式对话发生错误: {e}", exc_info=True)
 raise```
______________________________

## ...\clients\__init__.py
```python
from .base_client import BaseClient
from .deepseek_client import DeepSeekClient
from .claude_client import ClaudeClient
__all__ = ['BaseClient', 'DeepSeekClient', 'ClaudeClient']```
______________________________

## ...\deepclaude\deepclaude.py
```python
import json
import time
import tiktoken
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient
from app.utils.message_processor import MessageProcessor
import aiohttp
class DeepClaude:
 def __init__(
 self,
 deepseek_api_key: str,
 claude_api_key: str,
 deepseek_api_url: str = None,
 claude_api_url: str = "https://api.anthropic.com/v1/messages",
 claude_provider: str = "anthropic",
 is_origin_reasoning: bool = True
 ):
 self.deepseek_client = DeepSeekClient(
 api_key=deepseek_api_key,
 api_url=deepseek_api_url if deepseek_api_url else "https://api.siliconflow.cn/v1/chat/completions"
 )
 self.claude_client = ClaudeClient(
 api_key=claude_api_key,
 api_url=claude_api_url,
 provider=claude_provider
 )
 self.is_origin_reasoning = is_origin_reasoning
 self.retry_config = {
 'max_retries': 5,
 'base_delay': 2,
 'max_delay': 30
 }
 async def chat_completions_with_stream(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 deepseek_model: str = "deepseek-ai/DeepSeek-R1",
 claude_model: str = "claude-3-5-sonnet-20241022"
 ) -> AsyncGenerator[bytes, None]:
 if not messages:
 error_msg = "消息列表为空"
 logger.error(error_msg)
 raise ValueError(error_msg)
 for i in range(1, len(messages)):
 if messages[i].get("role") == messages[i-1].get("role"):
 error_msg = f"检测到连续的{messages[i].get('role')}消息"
 logger.warning(error_msg)
 raise ValueError(error_msg)
 message_processor = MessageProcessor()
 try:
 messages = message_processor.convert_to_deepseek_format(messages)
 logger.debug(f"转换后的消息: {messages}")
 except Exception as e:
 error_msg = f"消息格式转换失败: {str(e)}"
 logger.error(error_msg, exc_info=True)
 raise ValueError(error_msg)
 chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
 created_time = int(time.time())
 output_queue = asyncio.Queue()
 claude_queue = asyncio.Queue()
 reasoning_content = []
 async def process_deepseek():
 start_time = time.time()
 request_stats = {
 'retries': 0,
 'total_delay': 0,
 'errors': []
 }
 retry_count = 0
 max_retries = self.retry_config['max_retries']
 base_delay = self.retry_config['base_delay']
 while retry_count < max_retries:
 try:
 logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}, 提供商: {self.deepseek_client.provider}")
 start_response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": deepseek_model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": "🤔 思考过程:\n"
 }
 }]
 }
 await output_queue.put(f"data: {json.dumps(start_response)}\n\n".encode('utf-8'))
 async for content_type, content in self.deepseek_client.stream_chat(
 messages=messages,
 model=deepseek_model,
 is_origin_reasoning=self.is_origin_reasoning
 ):
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
 "content": content
 }
 }]
 }
 logger.debug(f"发送推理响应: {response}")
 await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
 elif content_type == "content":
 separator_response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": deepseek_model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": "\n\n---\n思考完毕，开始回答：\n\n"
 }
 }]
 }
 await output_queue.put(f"data: {json.dumps(separator_response)}\n\n".encode('utf-8'))
 logger.info(f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}")
 await claude_queue.put("".join(reasoning_content))
 break
 break
 except aiohttp.ClientError as e:
 retry_count += 1
 request_stats['retries'] += 1
 request_stats['errors'].append(str(e))
 if any(code in str(e) for code in ['504', '503', 'timeout']):
 delay = min(base_delay * (2 ** retry_count), self.retry_config['max_delay'])
 request_stats['total_delay'] += delay
 logger.warning(f"DeepSeek API 超时或服务不可用，第 {retry_count} 次重试，等待 {delay} 秒...")
 await asyncio.sleep(delay)
 continue
 if not any(code in str(e) for code in ['504', '503', 'timeout']):
 logger.error(f"处理 DeepSeek 流时发生错误: {e}", exc_info=True)
 await claude_queue.put("")
 break
 except asyncio.TimeoutError as e:
 retry_count += 1
 if retry_count < max_retries:
 delay = min(base_delay * (2 ** retry_count), 30)
 logger.warning(f"读取超时，第 {retry_count} 次重试，等待 {delay} 秒...")
 await asyncio.sleep(delay)
 continue
 logger.error(f"读取超时，达到最大重试次数: {e}", exc_info=True)
 await claude_queue.put("")
 break
 logger.info("DeepSeek 任务处理完成，标记结束")
 await output_queue.put(None)
 duration = time.time() - start_time
 logger.info(
 f"DeepSeek 请求统计:\n"
 f"总耗时: {duration:.2f}秒\n"
 f"重试次数: {request_stats['retries']}\n"
 f"总延迟: {request_stats['total_delay']}秒\n"
 f"错误记录: {request_stats['errors']}"
 )
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
 if not claude_messages:
 logger.error("Claude 消息列表为空")
 return
 claude_messages = [{
 "role": msg.get("role", "user"),
 "content": msg.get("content", "").strip()
 } for msg in claude_messages if msg.get("content", "").strip()]
 if not claude_messages:
 logger.error("处理后的 Claude 消息列表为空")
 return
 logger.debug(f"发送给 Claude 的消息: {claude_messages}")
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
 error_occurred = False
 while finished_tasks < 2:
 try:
 item = await output_queue.get()
 if item is None:
 finished_tasks += 1
 continue
 if isinstance(item, Exception):
 error_occurred = True
 logger.error(f"任务执行出错: {item}")
 continue
 logger.debug(f"自定义api向外发送 token: {item}")
 yield item
 except Exception as e:
 logger.error(f"处理输出队列时发生错误: {e}")
 error_occurred = True
 if error_occurred:
 error_response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": "error",
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": "\n\n抱歉，处理过程中出现错误，请稍后重试。"
 }
 }]
 }
 yield f"data: {json.dumps(error_response)}\n\n".encode('utf-8')
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
 if not api_key:
 logger.error("请在 .env 文件中设置 CLAUDE_API_KEY")
 return
 messages = [
 {"role": "user", "content": "1+1等于几?"}
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
 async for content_type, content in client.stream_chat(
 messages=messages,
 model="deepseek-ai/DeepSeek-R1",
 is_origin_reasoning=is_origin_reasoning
 ):
 if content_type == "reasoning":
 logger.info(f"收到推理内容: {content}")
 elif content_type == "content":
 logger.info(f"收到最终答案: {content}")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
def main():
 asyncio.run(test_deepseek_stream())
if __name__ == "__main__":
 main()```
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
# 声明容器将使用8211端口
# 这只是一个声明，实际运行时还需要通过-p参数映射端口
EXPOSE 8211

# 启动命令
# python -m uvicorn：通过Python模块的方式启动uvicorn服务器
# app.main:app：指定FastAPI应用的导入路径，格式为"模块路径:应用实例变量名"
# --host 0.0.0.0：允许来自任何IP的访问（不仅仅是localhost）
# --port 8211：指定服务器监听的端口号
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8211"]
```
______________________________
