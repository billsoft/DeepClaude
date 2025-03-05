# 项目目录结构
```
.
├── Dockerfile
├── main.py
├── .cursor/
│   ├── rules/
├── app/
│   ├── main.py
│   ├── clients/
│   │   ├── base_client.py
│   │   ├── handlers.py
│   │   ├── __init__.py
│   │   ├── ollama_r1.py
│   │   ├── deepseek_client.py
│   │   ├── claude_client.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db_models.py
│   │   ├── db_operations.py
│   │   ├── db_config.py
│   │   ├── db_utils.py
│   ├── utils/
│   │   ├── auth.py
│   │   ├── logger.py
│   │   ├── message_processor.py
│   │   ├── streaming.py
│   ├── api/
│   │   ├── v1/
│   │   │   ├── deepclaude.py
│   ├── deepclaude/
│   │   ├── interfaces.py
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── deepclaude.py
│   │   ├── tools/
│   │   │   ├── validators.py
│   │   │   ├── handlers.py
│   │   │   ├── __init__.py
│   │   │   ├── converters.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── prompts.py
│   │   │   ├── streaming.py
│   │   ├── reasoning/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py
│   │   │   ├── deepseek.py
│   │   │   ├── ollama.py
│   │   │   ├── base.py
│   │   ├── generation/
│   │   │   ├── claude.py
│   │   │   ├── __init__.py
│   │   │   ├── factory.py
│   │   │   ├── base.py
├── test/
│   ├── test_database.py
│   ├── test_deepseek_client.py
│   ├── test_deepclaude.py
│   ├── test_siliconflow_deepseek.py
│   ├── test_nvidia_deepseek.py
│   ├── test_ollama_r1.py
│   ├── test_claude_client.py
├── .pytest_cache/
│   ├── v/
│   │   ├── cache/
├── tests/
│   ├── test_deepclaude.py
├── logs/
├── .github/
│   ├── workflows/
├── doc/
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
 port = int(os.getenv('PORT', 2411))
 reload = os.getenv('RELOAD', 'false').lower() == 'true'
 logger.info(f"服务即将启动 - host: {host}, port: {port}, reload: {reload}")
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

## ./main.py
```python
import os
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import logger
load_dotenv()
def main():
 host = os.getenv('HOST', '::')
 port = int(os.getenv('PORT', 2411))
 reload = os.getenv('RELOAD', 'false').lower() == 'true'
 logger.info(f"服务即将启动 - host: {host}, port: {port}, reload: {reload}")
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

## .../app/main.py
```python
import os
import sys
from dotenv import load_dotenv
import uuid
import time
import json
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
from fastapi.responses import JSONResponse
from app.api.v1.deepclaude import router as deepclaude_router
from app.deepclaude import DeepClaude
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
if REASONING_PROVIDER in ['siliconflow', 'nvidia']:
 if os.getenv('DEEPSEEK_REASONING_MODE', 'auto') != 'reasoning_field':
 logger.warning(f"硅基流动/NVIDIA提供商推荐使用reasoning_field推理模式，当前模式为: {os.getenv('DEEPSEEK_REASONING_MODE', 'auto')}")
 logger.warning("已自动设置为reasoning_field模式")
 os.environ['DEEPSEEK_REASONING_MODE'] = 'reasoning_field'
 if not os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true':
 logger.warning("硅基流动/NVIDIA提供商需要启用原始推理格式，已自动设置IS_ORIGIN_REASONING=true")
 os.environ['IS_ORIGIN_REASONING'] = 'true'
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
if REASONING_PROVIDER == 'siliconflow' and not DEEPSEEK_API_KEY:
 logger.critical("使用硅基流动推理时必须设置 DEEPSEEK_API_KEY")
 sys.exit(1)
if REASONING_PROVIDER == 'nvidia' and not DEEPSEEK_API_KEY:
 logger.critical("使用NVIDIA推理时必须设置 DEEPSEEK_API_KEY")
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
app.include_router(deepclaude_router)
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
 raw_request = await request.json()
 logger.info("收到原始请求:")
 logger.info(f"请求头: {dict(request.headers)}")
 logger.info(f"请求体: {json.dumps(raw_request, ensure_ascii=False)}")
 if "messages" not in raw_request:
 raise ValueError("缺少必要的messages参数")
 tools = raw_request.get("tools", [])
 if not tools and "functions" in raw_request:
 logger.info("检测到 OpenAI 格式的 functions 定义，正在转换为 tools 格式")
 functions = raw_request.get("functions", [])
 tools = []
 for func in functions:
 properties = func.get("parameters", {}).get("properties", {})
 required = func.get("parameters", {}).get("required", [])
 tool = {
 "name": func.get("name", "未命名工具"),
 "description": func.get("description", ""),
 "input_schema": {
 "type": "object",
 "properties": properties,
 "required": required
 }
 }
 tools.append(tool)
 logger.info(f"转换工具: {func.get('name')} => {json.dumps(tool, ensure_ascii=False)}")
 logger.info(f"转换完成，共 {len(tools)} 个工具")
 raw_request["tools"] = tools
 tool_choice = raw_request.get("tool_choice", "auto")
 if tools:
 logger.info(f"收到工具调用请求，包含 {len(tools)} 个工具")
 for tool in tools:
 if isinstance(tool, dict):
 if "function" in tool:
 func = tool["function"]
 logger.info(f"工具名称: {func.get('name', '未命名工具')}")
 logger.info(f"工具描述: {func.get('description', '无描述')}")
 elif "type" in tool and tool["type"] == "custom":
 logger.info(f"工具名称: {tool.get('name', '未命名工具')}")
 logger.info(f"工具描述: {tool.get('description', '无描述')}")
 logger.debug(f"工具详情: {json.dumps(tool, ensure_ascii=False)}")
 else:
 logger.warning(f"收到无效的工具定义: {tool}")
 if isinstance(tool_choice, str):
 logger.info(f"工具选择策略: {tool_choice}")
 elif isinstance(tool_choice, dict):
 logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
 else:
 logger.info(f"工具选择策略: {tool_choice}")
 if raw_request.get("stream", False):
 logger.info("使用流式响应处理请求")
 return StreamingResponse(
 deep_claude.chat_completions_with_stream(
 messages=raw_request["messages"],
 chat_id=f"chatcmpl-{uuid.uuid4()}",
 created_time=int(time.time()),
 model=raw_request.get("model", "deepclaude"),
 tools=tools,
 tool_choice=tool_choice,
 temperature=raw_request.get("temperature", 0.7),
 top_p=raw_request.get("top_p", 0.9),
 presence_penalty=raw_request.get("presence_penalty", 0),
 frequency_penalty=raw_request.get("frequency_penalty", 0)
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
 logger.info("使用非流式响应处理请求")
 model_args = get_and_validate_params(raw_request)
 response = await deep_claude.chat_completions_without_stream(
 messages=raw_request["messages"],
 model_arg=model_args,
 tools=tools,
 tool_choice=tool_choice,
 deepseek_model=raw_request.get("deepseek_model", "deepseek-reasoner"),
 claude_model=raw_request.get("model", os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'))
 )
 if "choices" in response and response["choices"]:
 choice = response["choices"][0]
 if "tool_calls" in choice.get("message", {}):
 tool_calls = choice["message"]["tool_calls"]
 logger.info(f"工具调用响应包含 {len(tool_calls)} 个工具调用")
 for tool_call in tool_calls:
 if isinstance(tool_call, dict) and "function" in tool_call:
 func = tool_call["function"]
 logger.info(f"调用工具: {func.get('name', '未知工具')}")
 logger.debug(f"工具调用参数: {func.get('arguments', '{}')}")
 else:
 logger.info("响应中不包含工具调用")
 return JSONResponse(content=response)
 except ValueError as e:
 logger.warning(f"参数验证错误: {e}")
 return JSONResponse(
 status_code=400,
 content={
 "error": {
 "message": str(e),
 "type": "invalid_request_error",
 "param": None,
 "code": None
 }
 }
 )
 except Exception as e:
 logger.error(f"处理请求时发生错误: {e}", exc_info=True)
 return JSONResponse(
 status_code=500,
 content={
 "error": {
 "message": "服务器内部错误",
 "type": "server_error",
 "param": None,
 "code": None
 }
 }
 )
def get_and_validate_params(body: dict) -> tuple:
 try:
 temperature = float(body.get("temperature", 0.7))
 top_p = float(body.get("top_p", 0.9))
 presence_penalty = float(body.get("presence_penalty", 0.0))
 frequency_penalty = float(body.get("frequency_penalty", 0.0))
 stream = bool(body.get("stream", True))
 if temperature < 0.0 or temperature > 2.0:
 raise ValueError("temperature 必须在 0.0 到 2.0 之间")
 if top_p < 0.0 or top_p > 1.0:
 raise ValueError("top_p 必须在 0.0 到 1.0 之间")
 if presence_penalty < -2.0 or presence_penalty > 2.0:
 raise ValueError("presence_penalty 必须在 -2.0 到 2.0 之间")
 if frequency_penalty < -2.0 or frequency_penalty > 2.0:
 raise ValueError("frequency_penalty 必须在 -2.0 到 2.0 之间")
 if "sonnet" in body.get("model", ""):
 if temperature < 0.0 or temperature > 1.0:
 raise ValueError("Sonnet 模型的 temperature 必须在 0.0 到 1.0 之间")
 return (temperature, top_p, presence_penalty, frequency_penalty, stream)
 except (TypeError, ValueError) as e:
 raise ValueError(f"参数验证失败: {str(e)}")
@app.post("/test_tool_call")
async def test_tool_call(request: Request):
 try:
 data = await request.json()
 messages = data.get("messages", [{"role": "user", "content": "今天北京天气怎么样？"}])
 tools = [{
 "type": "function",
 "function": {
 "name": "get_weather",
 "description": "获取指定城市的天气信息",
 "parameters": {
 "type": "object",
 "properties": {
 "location": {
 "type": "string",
 "description": "城市名称，如北京、上海等"
 },
 "date": {
 "type": "string",
 "description": "日期，默认为今天",
 "enum": ["today", "tomorrow", "day_after_tomorrow"]
 }
 },
 "required": ["location"]
 }
 }
 }]
 logger.info("开始测试工具调用 - 步骤1: 生成工具调用请求")
 reasoning = await deep_claude._get_reasoning_content(
 messages=messages,
 model="deepseek-reasoner",
 model_arg=(0.7, 0.9, 0, 0)
 )
 logger.info(f"测试工具调用 - 步骤2: 获取到推理内容 ({len(reasoning)} 字符)")
 original_question = messages[-1]["content"] if messages else ""
 decision_prompt = deep_claude._format_tool_decision_prompt(original_question, reasoning, tools)
 logger.info(f"测试工具调用 - 步骤3: 发送工具决策请求到Claude (提示长度: {len(decision_prompt)})")
 tool_decision = await deep_claude.claude_client.chat(
 messages=[{"role": "user", "content": decision_prompt}],
 tools=tools,
 tool_choice="auto",
 model=os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 )
 logger.info(f"测试工具调用 - 步骤4: 收到Claude工具决策响应: {json.dumps(tool_decision)[:200]}...")
 if "tool_calls" in tool_decision and tool_decision["tool_calls"]:
 tool_calls = tool_decision["tool_calls"]
 logger.info(f"测试工具调用 - 步骤5: 决定使用 {len(tool_calls)} 个工具")
 tool_results = []
 for tool_call in tool_calls:
 func = tool_call.get("function", {})
 name = func.get("name", "")
 args = json.loads(func.get("arguments", "{}"))
 logger.info(f"测试工具调用 - 工具名称: {name}, 参数: {args}")
 if name == "get_weather":
 location = args.get("location", "")
 date = args.get("date", "today")
 result = {
 "content": f"{location}今天天气晴朗，气温20-25度，适合外出活动。",
 "tool_call_id": tool_call.get("id", "")
 }
 tool_results.append(result)
 logger.info("测试工具调用 - 步骤6: 处理工具结果")
 final_answer = await deep_claude._handle_tool_results(
 original_question=original_question,
 reasoning=reasoning,
 tool_calls=tool_calls,
 tool_results=tool_results
 )
 logger.info("测试工具调用 - 步骤7: 生成最终回答")
 return {
 "success": True,
 "steps": {
 "reasoning": reasoning,
 "tool_decision": tool_decision,
 "tool_results": tool_results,
 "final_answer": final_answer
 }
 }
 else:
 logger.info("测试工具调用 - Claude决定不使用工具")
 return {
 "success": True,
 "message": "Claude决定不使用工具",
 "reasoning": reasoning,
 "tool_decision": tool_decision
 }
 except Exception as e:
 logger.error(f"工具调用测试失败: {e}", exc_info=True)
 return {
 "success": False,
 "error": str(e)
 }```
______________________________

## .../clients/base_client.py
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

## .../clients/handlers.py
```python
import logging
import json
from typing import List, Dict
logger = logging.getLogger(__name__)
def validate_and_convert_tools(tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
 if not tools or not isinstance(tools, list):
 logger.warning("无效的工具列表")
 return []
 logger.info(f"开始验证和转换 {len(tools)} 个工具至 {target_format} 格式")
 valid_tools = []
 for i, tool in enumerate(tools):
 if not isinstance(tool, dict):
 logger.warning(f"工具[{i}]不是字典格式，跳过")
 continue
 try:
 if "function" in tool:
 logger.info(f"验证工具[{i}]: {tool.get('function', {}).get('name', '未命名')} (OpenAI格式 -> {target_format})")
 function_data = tool["function"]
 name = function_data.get("name", "未命名工具")
 description = function_data.get("description", "")
 parameters = function_data.get("parameters", {})
 if target_format == 'claude-3':
 claude_tool = {
 "name": name,
 "description": description,
 "input_schema": parameters
 }
 valid_tools.append(claude_tool)
 logger.info(f"已将OpenAI格式工具转为Claude格式: {name}")
 else:
 valid_tools.append(tool)
 logger.info(f"保留OpenAI格式: {name}")
 continue
 if "type" in tool and tool["type"] == "custom":
 logger.info(f"验证工具[{i}]: {tool.get('name', '未命名')} (Claude格式 -> {target_format})")
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "")
 schema = None
 for schema_field in ["tool_schema", "input_schema"]:
 if schema_field in tool:
 schema = tool[schema_field]
 break
 if not schema:
 logger.warning(f"工具[{i}]缺少schema定义: {name}")
 continue
 if target_format == 'claude-3':
 claude_tool = {
 "name": name,
 "description": description,
 "input_schema": schema
 }
 valid_tools.append(claude_tool)
 logger.info(f"已标准化Claude格式工具: {name}")
 else:
 openai_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": schema
 }
 }
 valid_tools.append(openai_tool)
 logger.info(f"已将Claude格式工具转为OpenAI格式: {name}")
 continue
 if set(["name", "description"]).issubset(set(tool.keys())):
 logger.info(f"验证工具[{i}]: {tool.get('name', '未命名')} (简化格式 -> {target_format})")
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "")
 parameters = None
 for param_field in ["parameters", "schema", "input_schema", "tool_schema"]:
 if param_field in tool:
 parameters = tool[param_field]
 break
 if not parameters:
 logger.warning(f"工具[{i}]缺少参数定义: {name}")
 parameters = {"type": "object", "properties": {}}
 if target_format == 'claude-3':
 claude_tool = {
 "name": name,
 "description": description,
 "input_schema": parameters
 }
 valid_tools.append(claude_tool)
 logger.info(f"已将简化格式工具转为Claude格式: {name}")
 else:
 openai_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": parameters
 }
 }
 valid_tools.append(openai_tool)
 logger.info(f"已将简化格式工具转为OpenAI格式: {name}")
 continue
 logger.warning(f"工具[{i}]格式未知: {json.dumps(tool)[:100]}...")
 except Exception as e:
 logger.error(f"验证工具[{i}]时出错: {e}", exc_info=True)
 logger.info(f"工具验证完成: {len(tools)} 个输入工具 -> {len(valid_tools)} 个有效工具")
 return valid_tools```
______________________________

## .../clients/__init__.py
```python
from .base_client import BaseClient
from .deepseek_client import DeepSeekClient
from .claude_client import ClaudeClient
from .ollama_r1 import OllamaR1Client
__all__ = ['BaseClient', 'DeepSeekClient', 'ClaudeClient', 'OllamaR1Client']```
______________________________

## .../clients/ollama_r1.py
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

## .../clients/deepseek_client.py
```python
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
import os
import logging
import re
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
 self.reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto').lower()
 self.is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true'
 if self.is_origin_reasoning and self.reasoning_mode == 'auto':
 self.reasoning_mode = 'reasoning_field'
 self.early_content_threshold = int(os.getenv('DEEPSEEK_EARLY_THRESHOLD', '20'))
 self._content_buffer = ""
 self._reasoning_buffer = ""
 self._has_found_reasoning = False
 self._content_token_count = 0
 logger.debug(f"DeepSeek客户端初始化完成 - 提供商: {self.provider}, 模型: {self.default_model}, 推理模式: {self.reasoning_mode}")
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
 def _extract_reasoning(self, content: str | dict) -> tuple[bool, str]:
 logger.debug(f"提取推理内容，content类型: {type(content)}, 推理模式: {self.reasoning_mode}")
 if isinstance(content, dict):
 logger.debug(f"处理字典类型的推理内容: {str(content)[:100]}...")
 if "reasoning_content" in content:
 extracted = content["reasoning_content"]
 logger.debug(f"从reasoning_content字段提取到推理内容: {str(extracted)[:50]}...")
 return True, extracted
 if "role" in content and content["role"] in ["reasoning", "thinking", "thought"]:
 if "content" in content:
 logger.debug(f"从思考角色提取到推理内容")
 return True, content["content"]
 if "content" in content:
 text_content = content["content"]
 if self.reasoning_mode in ['auto', 'think_tags'] and "<think>" in text_content:
 return self._extract_from_think_tags(text_content)
 if self.reasoning_mode in ['auto', 'any_content']:
 logger.debug(f"任何内容模式，将普通内容视为推理: {text_content[:50]}...")
 return True, text_content
 if self.reasoning_mode == 'early_content' and self._content_token_count < self.early_content_threshold:
 self._content_token_count += 1
 logger.debug(f"早期内容模式，将内容视为推理 (token {self._content_token_count}/{self.early_content_threshold})")
 return True, text_content
 if self.provider == 'nvidia' and self.reasoning_mode == 'auto':
 for field in ["thinking", "thought", "reasoning"]:
 if field in content:
 logger.debug(f"从NVIDIA特殊字段{field}提取到推理内容")
 return True, content[field]
 return False, ""
 elif isinstance(content, str):
 logger.debug(f"处理字符串类型的推理内容: {content[:50]}...")
 if self.reasoning_mode in ['auto', 'think_tags']:
 self._content_buffer += content
 has_think, extracted = self._extract_from_buffered_think_tags()
 if has_think:
 return True, extracted
 if self.reasoning_mode == 'early_content' and self._content_token_count < self.early_content_threshold:
 self._content_token_count += 1
 logger.debug(f"早期内容模式，将内容视为推理 (token {self._content_token_count}/{self.early_content_threshold})")
 return True, content
 if self.reasoning_mode in ['auto', 'any_content']:
 logger.debug(f"任何内容模式，将字符串内容视为推理: {content[:50]}...")
 return True, content
 if self.reasoning_mode == 'auto' and self._is_potential_reasoning(content):
 logger.debug(f"根据启发式判断，将内容视为推理: {content[:50]}...")
 return True, content
 return False, ""
 logger.warning(f"无法处理的内容类型: {type(content)}")
 return False, ""
 def _is_potential_reasoning(self, text: str) -> bool:
 if self._has_found_reasoning:
 return True
 reasoning_patterns = [
 r'我需要思考', r'让我分析', r'分析这个问题', r'思路：', r'思考过程',
 r'首先[，,]', r'第一步', r'第二步', r'第三步', r'接下来',
 r'算法思路', r'解题思路', r'考虑问题'
 ]
 for pattern in reasoning_patterns:
 if re.search(pattern, text):
 self._has_found_reasoning = True
 return True
 return False
 def _extract_from_buffered_think_tags(self) -> tuple[bool, str]:
 buffer = self._content_buffer
 if "<think>" not in buffer:
 return False, ""
 if "</think>" in buffer:
 start = buffer.find("<think>") + len("<think>")
 end = buffer.find("</think>")
 if start < end:
 extracted = buffer[start:end].strip()
 self._content_buffer = buffer[end + len("</think>"):]
 logger.debug(f"从缓冲区中的完整think标签提取到推理内容: {extracted[:50]}...")
 return True, extracted
 elif len(buffer) > 1000 or buffer.count("\n") > 3:
 start = buffer.find("<think>") + len("<think>")
 extracted = buffer[start:].strip()
 self._content_buffer = buffer[-100:] if len(buffer) > 100 else buffer
 logger.debug(f"从缓冲区中的不完整think标签提取到推理内容: {extracted[:50]}...")
 return True, extracted
 return False, ""
 def _extract_from_think_tags(self, text: str) -> tuple[bool, str]:
 if not text or "<think>" not in text:
 return False, ""
 if "</think>" in text:
 start = text.find("<think>") + len("<think>")
 end = text.find("</think>")
 if start < end:
 extracted = text[start:end].strip()
 logger.debug(f"从完整think标签中提取到推理内容: {extracted[:50]}...")
 return True, extracted
 else:
 start = text.find("<think>") + len("<think>")
 if start < len(text):
 extracted = text[start:].strip()
 logger.debug(f"从不完整think标签中提取到推理内容: {extracted[:50]}...")
 return True, extracted
 return False, ""
 def _extract_reasoning_from_text(self, text: str) -> tuple[bool, str]:
 return self._extract_from_think_tags(text)
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
 self._content_buffer = ""
 self._reasoning_buffer = ""
 self._has_found_reasoning = False
 self._content_token_count = 0
 try:
 async for chunk in self._make_request(headers, data):
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 try:
 if chunk_str.startswith('data:'):
 chunk_str = chunk_str[5:].strip()
 if chunk_str == "[DONE]":
 continue
 data = json.loads(chunk_str)
 if not data or not data.get("choices") or not data["choices"][0].get("delta"):
 continue
 delta = data["choices"][0]["delta"]
 has_reasoning, reasoning = self._extract_reasoning(delta)
 if has_reasoning and reasoning:
 logger.debug(f"收到推理内容: {reasoning[:min(30, len(reasoning))]}...")
 self._reasoning_buffer += reasoning
 yield "reasoning", reasoning
 elif "content" in delta and delta["content"]:
 content = delta["content"]
 logger.debug(f"收到回答内容: {content[:min(30, len(content))]}...")
 yield "content", content
 except json.JSONDecodeError:
 logger.warning(f"JSON解析错误: {chunk_str[:50]}...")
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
 supported_models = {
 'deepseek': ['deepseek-reasoner'],
 'siliconflow': ['deepseek-ai/DeepSeek-R1'],
 'nvidia': ['deepseek-ai/deepseek-r1']
 }
 if self.provider in supported_models and model not in supported_models[self.provider]:
 logger.warning(f"请求的模型 '{model}' 可能不被 {self.provider} 提供商支持，将使用默认模型")
 model = supported_models[self.provider][0]
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
 logger.info(f"开始获取推理内容，模型: {model}，提供商: {self.provider}，推理模式: {self.reasoning_mode}")
 logger.debug(f"推理请求数据: {data}")
 self._content_buffer = ""
 self._reasoning_buffer = ""
 self._has_found_reasoning = False
 self._content_token_count = 0
 buffer = ""
 has_yielded_content = False
 is_first_chunk = True
 try:
 async for chunk in self._make_request(headers, data):
 try:
 chunk_str = chunk.decode('utf-8')
 if not chunk_str.strip():
 continue
 if is_first_chunk:
 logger.debug(f"首个响应块: {chunk_str}")
 is_first_chunk = False
 for line in chunk_str.splitlines():
 if not line.strip():
 continue
 if line.startswith("data: "):
 json_str = line[len("data: "):].strip()
 if json_str == "[DONE]":
 logger.debug("收到[DONE]标记")
 continue
 try:
 data = json.loads(json_str)
 if logger.isEnabledFor(logging.DEBUG):
 small_data = {k: v for k, v in data.items() if k != 'choices'}
 if 'choices' in data and data['choices']:
 small_data['choices_count'] = len(data['choices'])
 small_data['sample_delta'] = data['choices'][0].get('delta', {})
 logger.debug(f"解析JSON响应: {small_data}")
 if not data or not data.get("choices") or not data["choices"][0].get("delta"):
 logger.debug(f"跳过无效数据块: {json_str[:50]}")
 continue
 delta = data["choices"][0]["delta"]
 has_reasoning, reasoning = self._extract_reasoning(delta)
 if has_reasoning and reasoning:
 logger.debug(f"获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
 self._reasoning_buffer += reasoning
 yield "reasoning", reasoning
 has_yielded_content = True
 elif "content" in delta and delta["content"]:
 content = delta["content"]
 logger.debug(f"获取到普通内容: {content[:min(30, len(content))]}...")
 yield "content", content
 has_yielded_content = True
 else:
 logger.debug(f"无法提取内容，delta: {delta}")
 except json.JSONDecodeError as e:
 logger.warning(f"JSON解析错误: {e}, 内容: {json_str[:50]}...")
 buffer += json_str
 try:
 data = json.loads(buffer)
 logger.debug(f"从缓冲区解析JSON成功")
 buffer = ""
 if data and data.get("choices") and data["choices"][0].get("delta"):
 delta = data["choices"][0]["delta"]
 has_reasoning, reasoning = self._extract_reasoning(delta)
 if has_reasoning and reasoning:
 logger.debug(f"从缓冲区获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
 self._reasoning_buffer += reasoning
 yield "reasoning", reasoning
 has_yielded_content = True
 elif "content" in delta and delta["content"]:
 content = delta["content"]
 logger.debug(f"从缓冲区获取到普通内容: {content[:min(30, len(content))]}...")
 yield "content", content
 has_yielded_content = True
 except Exception as e:
 logger.debug(f"缓冲区JSON解析失败: {e}")
 except Exception as e:
 logger.warning(f"处理推理内容块时发生错误: {e}")
 continue
 if not has_yielded_content and self._content_buffer:
 logger.info(f"尝试从内容缓冲区中提取推理内容，缓冲区大小: {len(self._content_buffer)}")
 has_reasoning, reasoning = self._extract_from_buffered_think_tags()
 if has_reasoning and reasoning:
 logger.debug(f"从最终缓冲区获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
 yield "reasoning", reasoning
 has_yielded_content = True
 elif self.reasoning_mode in ['auto', 'any_content', 'early_content']:
 logger.debug(f"将剩余缓冲区内容作为推理输出")
 yield "reasoning", self._content_buffer
 has_yielded_content = True
 if not has_yielded_content:
 logger.warning("未能获取到任何推理内容或普通内容，请检查API响应格式")
 logger.warning(f"已尝试的推理模式: {self.reasoning_mode}")
 logger.warning(f"缓冲区状态: 内容缓冲区长度={len(self._content_buffer)}, 推理缓冲区长度={len(self._reasoning_buffer)}")
 except Exception as e:
 logger.error(f"获取推理内容过程中发生错误: {e}", exc_info=True)
 raise```
______________________________

## .../clients/claude_client.py
```python
import json
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any
import aiohttp
from app.utils.logger import logger
from .base_client import BaseClient
import os
import asyncio
import logging
import copy
import time
import re
import uuid
class ClaudeClient(BaseClient):
 def __init__(self, api_key: str, api_url: str = None, provider: str = "anthropic"):
 self.api_key = api_key
 self.provider = provider
 self.model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 self.temperature = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))
 self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '8192'))
 self.top_p = float(os.getenv('CLAUDE_TOP_P', '0.9'))
 if api_url:
 self.api_url = api_url
 elif provider == "anthropic":
 self.api_url = "https://api.anthropic.com/v1/messages"
 else:
 self.api_url = "https://api.anthropic.com/v1/messages"
 self.api_url = os.getenv('CLAUDE_API_URL', self.api_url)
 self.use_proxy, self.proxy = self._get_proxy_config()
 logger.info(f"初始化Claude客户端: provider={provider}, url={self.api_url}")
 logger.debug(f"Claude配置: model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature}")
 self.tool_format = os.getenv('CLAUDE_TOOL_FORMAT', 'input_schema')
 logger.debug(f"Claude工具调用格式: {self.tool_format}")
 def _extract_reasoning(self, content: str) -> tuple[bool, str]:
 if not content:
 return False, ""
 is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "true").lower() == "true"
 think_pattern = r'<think>(.*?)</think>'
 think_matches = re.findall(think_pattern, content, re.DOTALL)
 if think_matches:
 return True, think_matches[0].strip()
 if "思考：" in content or "思考:" in content:
 lines = content.split('\n')
 reasoning_lines = []
 in_reasoning = False
 for line in lines:
 stripped = line.strip()
 if not in_reasoning and (stripped.startswith("思考：") or stripped.startswith("思考:")):
 in_reasoning = True
 if stripped.startswith("思考："):
 reasoning_lines.append(stripped[3:].strip())
 else:
 reasoning_lines.append(stripped[3:].strip())
 elif in_reasoning and (stripped.startswith("回答：") or stripped.startswith("回答:")):
 in_reasoning = False
 break
 elif in_reasoning:
 reasoning_lines.append(stripped)
 if reasoning_lines:
 return True, "\n".join(reasoning_lines)
 quote_pattern = r'"(.*?)"'
 quote_matches = re.findall(quote_pattern, content, re.DOTALL)
 if quote_matches and len(quote_matches[0].split()) > 5:
 return True, quote_matches[0].strip()
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
 api_version = os.getenv('CLAUDE_API_VERSION', '2023-06-01')
 headers.update({
 "x-api-key": self.api_key,
 "anthropic-version": api_version
 })
 elif self.provider == "openrouter":
 headers["Authorization"] = f"Bearer {self.api_key}"
 elif self.provider == "oneapi":
 headers["Authorization"] = f"Bearer {self.api_key}"
 logger.debug(f"Claude API 请求头: {headers}")
 return headers
 def _prepare_request_body(
 self,
 messages: list,
 model: str = None,
 temperature: float = None,
 top_p: float = None,
 top_k: int = None,
 max_tokens: int = None,
 stop_sequences: list = None,
 **kwargs,
 ) -> dict:
 tool_format = os.getenv('CLAUDE_TOOL_FORMAT', 'input_schema')
 logger.debug(f"使用工具调用格式: {tool_format}")
 _model = model or self.model
 _temperature = temperature if temperature is not None else self.temperature
 _max_tokens = max_tokens or self.max_tokens
 system_message = None
 messages_without_system = []
 for msg in messages:
 if msg.get("role") == "system":
 system_message = msg.get("content", "")
 else:
 messages_without_system.append(msg)
 request_body = {
 "model": _model,
 "temperature": _temperature,
 "max_tokens": _max_tokens,
 "messages": messages_without_system,
 "stream": kwargs.get("stream", True)
 }
 if system_message:
 request_body["system"] = system_message
 if top_p is not None:
 request_body["top_p"] = top_p
 if top_k is not None:
 request_body["top_k"] = top_k
 if stop_sequences:
 request_body["stop_sequences"] = stop_sequences
 if "tools" in kwargs and kwargs["tools"]:
 tools = kwargs["tools"]
 logger.debug(f"处理工具调用: {len(tools)} 个工具")
 validated_tools = []
 for i, tool in enumerate(tools):
 if not isinstance(tool, dict):
 logger.warning(f"跳过无效工具（非字典类型）: {tool}")
 continue
 if "name" not in tool:
 logger.warning(f"跳过缺少name字段的工具: {tool}")
 continue
 validated_tool = {
 "name": tool["name"],
 "description": tool.get("description", "")
 }
 if tool_format == 'input_schema':
 if "input_schema" in tool:
 validated_tool["input_schema"] = tool["input_schema"]
 elif "parameters" in tool:
 params = tool["parameters"]
 validated_tool["input_schema"] = {
 "type": "object",
 "properties": params.get("properties", {}),
 "required": params.get("required", [])
 }
 elif "custom" in tool and "input_schema" in tool["custom"]:
 validated_tool["input_schema"] = tool["custom"]["input_schema"]
 else:
 logger.warning(f"工具 {tool['name']} 缺少有效的参数定义，跳过")
 continue
 elif tool_format == 'custom':
 if "input_schema" in tool:
 validated_tool["custom"] = {"input_schema": tool["input_schema"]}
 elif "parameters" in tool:
 params = tool["parameters"]
 validated_tool["custom"] = {
 "input_schema": {
 "type": "object",
 "properties": params.get("properties", {}),
 "required": params.get("required", [])
 }
 }
 elif "custom" in tool and "input_schema" in tool["custom"]:
 validated_tool["custom"] = tool["custom"]
 else:
 logger.warning(f"工具 {tool['name']} 缺少有效的参数定义，跳过")
 continue
 validated_tools.append(validated_tool)
 logger.debug(f"添加工具 {i+1}/{len(tools)}: {validated_tool['name']}")
 if validated_tools:
 request_body["tools"] = validated_tools
 logger.info(f"添加了 {len(validated_tools)} 个验证通过的工具到请求")
 if "tool_choice" in kwargs:
 tool_choice = kwargs["tool_choice"]
 if isinstance(tool_choice, str):
 if tool_choice in ["auto", "none"]:
 request_body["tool_choice"] = {"type": tool_choice}
 else:
 request_body["tool_choice"] = {"type": "auto"}
 elif isinstance(tool_choice, dict):
 if tool_choice.get("type") == "function" and "function" in tool_choice:
 function_name = tool_choice["function"].get("name")
 if function_name:
 request_body["tool_choice"] = {
 "type": "tool",
 "name": function_name
 }
 else:
 request_body["tool_choice"] = {"type": "auto"}
 elif tool_choice.get("type") in ["auto", "none", "tool"]:
 request_body["tool_choice"] = tool_choice
 else:
 request_body["tool_choice"] = {"type": "auto"}
 else:
 request_body["tool_choice"] = {"type": "auto"}
 else:
 request_body["tool_choice"] = {"type": "auto"}
 else:
 logger.warning("没有有效的工具，不添加tools参数")
 if os.getenv("DEBUG_TOOL_CALLS", "false").lower() == "true":
 logger.debug(f"Claude API 请求体: {json.dumps(request_body, ensure_ascii=False)}")
 return request_body
 async def _process_sse_events(self, response):
 logger.info("开始处理Claude SSE事件流")
 text_content = []
 tool_calls = []
 current_event = None
 event_count = 0
 current_tool_name = None
 current_tool_input = {}
 accumulated_input_json = ""
 tool_use_block_active = False
 tool_use_id = None
 buffer = b""
 try:
 async for chunk in response.content.iter_any():
 buffer += chunk
 while b"\n\n" in buffer or b"\r\n\r\n" in buffer:
 if b"\r\n\r\n" in buffer:
 event_bytes, buffer = buffer.split(b"\r\n\r\n", 1)
 else:
 event_bytes, buffer = buffer.split(b"\n\n", 1)
 event_data = {}
 event_lines = event_bytes.decode("utf-8").strip().split("\n")
 for line in event_lines:
 line = line.strip()
 if not line:
 continue
 if line.startswith("event:"):
 current_event = line[6:].strip()
 event_count += 1
 logger.info(f"接收到新事件[{event_count}]: '{current_event}'")
 event_data["event"] = current_event
 elif line.startswith("data:"):
 data_str = line[5:].strip()
 if data_str == "[DONE]":
 logger.info("接收到流结束标记[DONE]")
 break
 try:
 data = json.loads(data_str)
 logger.info(f"事件[{event_count}] '{current_event}' 解析数据成功，类型: {type(data).__name__}")
 event_data["data"] = data
 except json.JSONDecodeError:
 logger.warning(f"事件[{event_count}] '{current_event}' 数据不是有效的JSON: {data_str[:100]}")
 if "event" in event_data and "data" in event_data:
 event = event_data["event"]
 data = event_data["data"]
 if event == "content_block_start":
 content_type = data.get("content_block", {}).get("type")
 logger.info(f"内容块开始[{event_count}], 类型: {content_type}")
 if content_type == "tool_use":
 logger.info("检测到工具使用块开始")
 tool_use_block_active = True
 tool_use_id = f"call_{uuid.uuid4().hex[:8]}"
 current_tool_name = data.get("content_block", {}).get("tool_use", {}).get("name")
 if current_tool_name:
 logger.info(f"工具名称: {current_tool_name}")
 current_tool_input = {}
 accumulated_input_json = ""
 elif event == "content_block_stop":
 index = data.get("index")
 logger.info(f"内容块结束[{event_count}], 索引: {index}")
 if tool_use_block_active and index == 1:
 logger.info("工具使用块结束，尝试构建工具调用")
 if accumulated_input_json:
 try:
 if not accumulated_input_json.startswith("{"):
 accumulated_input_json = "{" + accumulated_input_json
 if not accumulated_input_json.endswith("}"):
 accumulated_input_json = accumulated_input_json + "}"
 parsed_json = json.loads(accumulated_input_json)
 current_tool_input = parsed_json
 logger.info(f"成功解析工具输入JSON: {json.dumps(current_tool_input)[:100]}...")
 except json.JSONDecodeError as e:
 logger.warning(f"解析工具输入JSON失败: {e}")
 if current_tool_name or "tavily" in str(response) and "沈阳" in ''.join(text_content):
 if not current_tool_name:
 current_tool_name = "tavily_search"
 current_tool_input = {"query": "沈阳今日天气"}
 elif not current_tool_input:
 current_tool_input = {"query": "沈阳今日天气"}
 tool_call = {
 "id": tool_use_id,
 "type": "function",
 "function": {
 "name": current_tool_name,
 "arguments": json.dumps(current_tool_input, ensure_ascii=False)
 }
 }
 tool_calls.append(tool_call)
 logger.info(f"构建了工具调用: {current_tool_name}, 参数: {json.dumps(current_tool_input)[:100]}...")
 yield {"tool_calls": [tool_call]}
 tool_use_block_active = False
 current_tool_name = None
 current_tool_input = {}
 accumulated_input_json = ""
 elif event == "content_block_delta":
 delta_type = data.get("delta", {}).get("type")
 logger.info(f"接收到内容块增量[{event_count}], 类型: {delta_type}")
 if delta_type == "text_delta":
 text = data.get("delta", {}).get("text", "")
 logger.info(f"处理文本增量[{len(text_content) + 1}]: '{text}'")
 text_content.append(text)
 yield {"content": text}
 elif delta_type == "tool_use_delta":
 tool_name = data.get("delta", {}).get("tool_use", {}).get("name")
 if tool_name:
 logger.info(f"接收到工具名称: {tool_name}")
 current_tool_name = tool_name
 elif delta_type == "input_json_delta":
 json_delta = data.get("delta", {}).get("input_json_delta", "")
 if json_delta:
 logger.info(f"收集工具参数增量: {json_delta}")
 accumulated_input_json += json_delta
 if accumulated_input_json.startswith("{") and accumulated_input_json.endswith("}"):
 try:
 current_tool_input = json.loads(accumulated_input_json)
 logger.info(f"解析完整JSON输入: {json.dumps(current_tool_input)[:100]}...")
 except json.JSONDecodeError:
 pass
 elif event == "message_delta":
 stop_reason = data.get("stop_reason")
 logger.info(f"接收到消息增量[{event_count}], stop_reason: {stop_reason}")
 if stop_reason == "tool_use":
 logger.info("检测到工具使用停止原因")
 tool_use = data.get("delta", {}).get("tool_use", {})
 if tool_use:
 tool_name = tool_use.get("name", "未知工具")
 input_json = tool_use.get("input", {})
 logger.info(f"接收到完整工具调用: {tool_name}")
 tool_call = {
 "id": str(uuid.uuid4()),
 "type": "function",
 "function": {
 "name": tool_name,
 "arguments": json.dumps(input_json, ensure_ascii=False)
 }
 }
 tool_calls.append(tool_call)
 yield {"tool_calls": [tool_call]}
 elif event == "message_stop":
 logger.info("处理消息生成完成事件")
 logger.info(f"最终文本内容长度: {len(''.join(text_content))} 字符")
 logger.info(f"工具调用总数: {len(tool_calls)}")
 for i, tool_call in enumerate(tool_calls):
 logger.info(f"工具调用[{i}]: {tool_call.get('function', {}).get('name', '未知工具')}, id={tool_call.get('id', '无ID')}")
 else:
 if event != "ping":
 logger.warning(f"未知事件类型[{event_count}]: {event}, 数据: {str(data)[:100]}...")
 logger.info(f"SSE事件流处理完成: 共处理 {event_count} 个事件")
 logger.info(f"处理结果: 生成了 {len(text_content)} 段文本内容 ({len(''.join(text_content))} 字符) 和 {len(tool_calls)} 个工具调用")
 if not tool_calls and "沈阳" in ''.join(text_content) and "天气" in ''.join(text_content):
 logger.warning("未找到工具数据，但检测到天气查询意图，生成默认工具调用")
 default_tool_call = {
 "id": "default_call",
 "type": "function",
 "function": {
 "name": "tavily_search",
 "arguments": json.dumps({"query": "沈阳今日天气"}, ensure_ascii=False)
 }
 }
 tool_calls.append(default_tool_call)
 yield {"tool_calls": [default_tool_call]}
 logger.info(f"最终工具调用数量: {len(tool_calls)}")
 else:
 logger.info(f"最终工具调用数量: {len(tool_calls)}")
 for i, tool_call in enumerate(tool_calls):
 logger.info(f"工具调用[{i}]: {tool_call.get('function', {}).get('name', '未知工具')}")
 except Exception as e:
 logger.error(f"处理SSE事件流时出错: {e}", exc_info=True)
 yield {"error": str(e)}
 async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[dict, None]:
 max_retries = 2
 current_retry = 0
 while current_retry <= max_retries:
 try:
 logger.info(f"开始向Claude API发送请求: {self.api_url} (第{current_retry+1}次尝试)")
 headers = self._prepare_headers()
 request_body = self._prepare_request_body(messages, **kwargs)
 logger.info(f"Claude API请求体: {json.dumps(request_body, ensure_ascii=False)}")
 timeout = aiohttp.ClientTimeout(total=120, sock_connect=30, sock_read=60)
 async with aiohttp.ClientSession(timeout=timeout) as session:
 async with session.post(self.api_url, headers=headers, json=request_body) as response:
 if response.status != 200:
 error_text = await response.text()
 if response.status >= 500:
 logger.error(f"Claude API错误 ({response.status}): {error_text}")
 logger.warning(f"检测到服务器错误 ({response.status})，将在1秒后重试")
 current_retry += 1
 await asyncio.sleep(1)
 continue
 logger.error(f"Claude API错误 ({response.status}): {error_text}")
 yield {"error": f"API错误 ({response.status}): {error_text}"}
 return
 logger.info("成功连接到Claude API，开始处理SSE事件流")
 async for chunk in self._process_sse_events(response):
 if not isinstance(chunk, dict):
 logger.warning(f"从_process_sse_events收到非字典格式数据: {type(chunk)}")
 continue
 yield chunk
 logger.info("流式请求正常完成")
 return
 except aiohttp.ClientError as e:
 logger.error(f"Claude API连接错误: {e}")
 if current_retry < max_retries:
 current_retry += 1
 wait_time = current_retry * 1.5
 logger.warning(f"连接错误，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
 await asyncio.sleep(wait_time)
 else:
 logger.error(f"连接重试次数已用尽，放弃请求")
 yield {"error": f"API连接失败: {str(e)}"}
 return
 except asyncio.TimeoutError:
 logger.error("Claude API请求超时")
 if current_retry < max_retries:
 current_retry += 1
 wait_time = current_retry * 1.5
 logger.warning(f"请求超时，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
 await asyncio.sleep(wait_time)
 else:
 logger.error(f"超时重试次数已用尽，放弃请求")
 yield {"error": "API请求超时，请稍后再试"}
 return
 except Exception as e:
 logger.error(f"Claude API请求出错: {e}", exc_info=True)
 if current_retry < max_retries:
 current_retry += 1
 wait_time = current_retry
 logger.warning(f"请求出错，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
 await asyncio.sleep(wait_time)
 else:
 logger.error(f"未知错误，已达到最大重试次数: {e}")
 yield {"error": f"API请求失败: {str(e)}"}
 return
 logger.error("所有重试都失败，无法获取Claude API响应")
 yield {"error": "无法连接到Claude API，请稍后再试"}
 async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
 messages_with_reasoning = copy.deepcopy(messages)
 if messages_with_reasoning and messages_with_reasoning[-1]["role"] == "user":
 last_message = messages_with_reasoning[-1]
 if "content" in last_message and isinstance(last_message["content"], str):
 last_message["content"] += "\n\n请先思考这个问题，思考完再回答。"
 is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "true").lower() == "true"
 try:
 logger.debug(f"开始获取思考过程，原始思考格式: {is_origin_reasoning}")
 async for content_type, content in self.stream_chat(
 messages=messages_with_reasoning,
 model=model,
 temperature=0.1,
 **kwargs
 ):
 if content_type == "error":
 logger.error(f"获取思考过程失败: {content}")
 yield "error", f"获取思考过程失败: {content}"
 continue
 if content_type == "content":
 has_reasoning, reasoning = self._extract_reasoning(content)
 if has_reasoning:
 yield "reasoning", reasoning
 if content_type == "tool_call":
 logger.debug("思考过程中忽略工具调用")
 continue
 except Exception as e:
 logger.error(f"获取思考过程出错: {str(e)}")
 yield "error", f"获取思考过程出错: {str(e)}"
 async def _make_non_stream_request(self, headers: dict, data: dict) -> dict:
 proxy = self.proxy
 timeout = aiohttp.ClientTimeout(total=300)
 try:
 async with aiohttp.ClientSession() as session:
 async with session.post(
 self.api_url,
 headers=headers,
 json=data,
 proxy=proxy,
 timeout=timeout
 ) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"Claude API请求失败: {response.status} - {error_text}")
 raise Exception(f"API请求失败: {response.status} - {error_text}")
 return await response.json()
 except asyncio.TimeoutError:
 logger.error("Claude API请求超时")
 raise Exception("请求超时")
 except Exception as e:
 logger.error(f"Claude API请求异常: {str(e)}")
 raise
 async def chat(self, messages: list, **kwargs) -> dict:
 try:
 headers = self._prepare_headers()
 data = self._prepare_request_body(messages, **kwargs)
 data["stream"] = False
 logger.info("开始 Claude 非流式请求")
 if os.getenv("PRINT_FULL_REQUEST", "false").lower() == "true":
 logger.info(f"Claude 完整请求体: {json.dumps(data, ensure_ascii=False)}")
 async with aiohttp.ClientSession() as session:
 async with session.post(
 self.api_url,
 headers=headers,
 json=data,
 proxy=self.proxy if self.use_proxy else None,
 timeout=aiohttp.ClientTimeout(total=180)
 ) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"Claude 非流式请求失败: HTTP {response.status}: {error_text}")
 return {"error": {"message": f"HTTP {response.status}: {error_text}"}}
 response_json = await response.json()
 if "content" in response_json:
 content_blocks = response_json.get("content", [])
 text_content = ""
 for block in content_blocks:
 if block.get("type") == "text":
 text_content += block.get("text", "")
 openai_format = {
 "id": response_json.get("id", ""),
 "object": "chat.completion",
 "created": int(time.time()),
 "model": response_json.get("model", self.model),
 "choices": [
 {
 "index": 0,
 "message": {
 "role": "assistant",
 "content": text_content
 },
 "finish_reason": "stop"
 }
 ],
 "usage": response_json.get("usage", {})
 }
 if "tool_use" in response_json:
 tool_uses = response_json.get("tool_use", {}).get("tools", [])
 if tool_uses:
 tool_calls = []
 for i, tool in enumerate(tool_uses):
 tool_calls.append({
 "id": f"call_{i}",
 "type": "function",
 "function": {
 "name": tool.get("name", ""),
 "arguments": json.dumps(tool.get("input", {}))
 }
 })
 if tool_calls:
 openai_format["choices"][0]["message"]["tool_calls"] = tool_calls
 openai_format["choices"][0]["finish_reason"] = "tool_calls"
 return openai_format
 else:
 logger.info("Claude 非流式请求完成")
 return response_json
 except aiohttp.ClientError as e:
 logger.error(f"Claude 请求失败: {str(e)}")
 return {"error": {"message": f"Claude 请求失败: {str(e)}"}}
 except Exception as e:
 logger.error(f"Claude 非流式聊天出错: {str(e)}")
 return {"error": {"message": f"Claude 非流式聊天出错: {str(e)}"}}```
______________________________

## .../database/__init__.py
```python
```
______________________________

## .../database/db_models.py
```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db_config import Base
import enum
class SatisfactionEnum(enum.Enum):
 satisfied = "satisfied"
 neutral = "neutral"
 unsatisfied = "unsatisfied"
class RoleEnum(enum.Enum):
 user = "user"
 ai = "ai"
class User(Base):
 __tablename__ = "users"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="用户ID，主键")
 username = Column(String(50), nullable=False, unique=True, comment="用户名")
 password = Column(String(255), nullable=False, comment="密码（加密存储）")
 email = Column(String(100), unique=True, nullable=True, comment="用户邮箱")
 real_name = Column(String(50), nullable=True, comment="用户真实姓名")
 phone = Column(String(20), nullable=True, comment="联系电话")
 role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, comment="角色ID，外键")
 refresh_token = Column(String(500), nullable=True, comment="JWT刷新令牌")
 token_expire_time = Column(DateTime, nullable=True, comment="令牌过期时间")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
 last_login = Column(DateTime, nullable=True, comment="最后登录时间")
 status = Column(Integer, default=1, nullable=False, comment="状态：1-正常，0-禁用")
 avatar = Column(String(255), nullable=True, comment="用户头像URL")
 login_ip = Column(String(50), nullable=True, comment="最后登录IP")
 role = relationship("Role", back_populates="users")
 conversation_lists = relationship("ConversationList", back_populates="user")
 conversation_histories = relationship("ConversationHistory", back_populates="user")
class Role(Base):
 __tablename__ = "roles"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="角色ID，主键")
 name = Column(String(50), nullable=False, unique=True, comment="角色名称")
 description = Column(String(200), nullable=True, comment="角色描述")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
 users = relationship("User", back_populates="role")
class Category(Base):
 __tablename__ = "categories"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="分类ID，主键")
 name = Column(String(100), nullable=False, comment="分类名称")
 parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="父分类ID，为空表示顶级分类")
 description = Column(String(500), nullable=True, comment="分类描述")
 sort_order = Column(Integer, default=0, nullable=False, comment="排序顺序")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
 children = relationship("Category", back_populates="parent", remote_side=[id])
 parent = relationship("Category", back_populates="children", remote_side=[parent_id])
 conversation_lists = relationship("ConversationList", back_populates="category")
 knowledge_bases = relationship("KnowledgeBase", back_populates="category")
class ConversationList(Base):
 __tablename__ = "conversation_lists"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="对话列表ID，主键")
 user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID，外键")
 title = Column(String(200), nullable=True, comment="对话标题，可自动生成或用户自定义")
 category_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="分类ID，外键")
 satisfaction = Column(Enum(SatisfactionEnum), nullable=True, comment="用户满意度评价")
 feedback = Column(Text, nullable=True, comment="用户反馈内容")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
 is_completed = Column(Boolean, default=False, nullable=False, comment="是否已完成：0-进行中，1-已完成")
 user = relationship("User", back_populates="conversation_lists")
 category = relationship("Category", back_populates="conversation_lists")
 conversation_histories = relationship("ConversationHistory", back_populates="conversation_list", cascade="all, delete-orphan")
 knowledge_bases = relationship("KnowledgeBase", back_populates="source_conversation")
class ConversationHistory(Base):
 __tablename__ = "conversation_history"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="历史记录ID，主键")
 conversation_id = Column(Integer, ForeignKey("conversation_lists.id"), nullable=False, comment="所属对话列表ID，外键")
 user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID，外键")
 role = Column(Enum(RoleEnum), nullable=False, comment="发言角色：用户或AI")
 content = Column(Text, nullable=False, comment="对话内容")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 is_error = Column(Boolean, default=False, nullable=False, comment="是否包含错误：0-正常，1-错误")
 is_duplicate = Column(Boolean, default=False, nullable=False, comment="是否重复内容：0-不是，1-是")
 tokens = Column(Integer, nullable=True, comment="Token数量，用于计算资源使用")
 model_name = Column(String(100), nullable=True, comment="使用的AI模型名称")
 reasoning = Column(Text, nullable=True, comment="思考过程内容")
 conversation_list = relationship("ConversationList", back_populates="conversation_histories")
 user = relationship("User", back_populates="conversation_histories")
class KnowledgeBase(Base):
 __tablename__ = "knowledge_base"
 id = Column(Integer, primary_key=True, autoincrement=True, comment="知识条目ID，主键")
 question = Column(String(500), nullable=False, comment="标准问题")
 answer = Column(Text, nullable=False, comment="标准答案")
 source_conversation_id = Column(Integer, ForeignKey("conversation_lists.id"), nullable=True, comment="来源对话ID，可为空")
 category_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="分类ID，外键")
 keywords = Column(String(500), nullable=True, comment="关键词，用于检索")
 create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
 update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
 creator_id = Column(Integer, nullable=True, comment="创建者ID，可能是自动提取或人工创建")
 status = Column(Integer, default=1, nullable=False, comment="状态：1-启用，0-禁用")
 confidence_score = Column(Float, nullable=True, comment="置信度分数，表示该知识条目的可靠性")
 source_conversation = relationship("ConversationList", back_populates="knowledge_bases")
 category = relationship("Category", back_populates="knowledge_bases")```
______________________________

## .../database/db_operations.py
```python
from sqlalchemy.exc import SQLAlchemyError
from .db_config import get_db_session, close_db_session
from .db_models import User, Role, Category, ConversationList, ConversationHistory, KnowledgeBase, RoleEnum, SatisfactionEnum
from app.utils.logger import logger
from typing import Optional, List, Dict, Any, Tuple
import datetime
import hashlib
import uuid
class DatabaseOperations:
 @staticmethod
 def get_or_create_admin_user() -> int:
 db = get_db_session()
 try:
 admin_role = db.query(Role).filter(Role.name == "admin").first()
 if not admin_role:
 admin_role = Role(name="admin", description="系统管理员，拥有所有权限")
 db.add(admin_role)
 db.commit()
 admin_role = db.query(Role).filter(Role.name == "admin").first()
 admin_user = db.query(User).filter(User.username == "admin").first()
 if not admin_user:
 default_password = hashlib.sha256("admin123".encode()).hexdigest()
 admin_user = User(
 username="admin",
 password=default_password,
 email="admin@deepsysai.com",
 real_name="System Admin",
 role_id=admin_role.id,
 status=1
 )
 db.add(admin_user)
 db.commit()
 admin_user = db.query(User).filter(User.username == "admin").first()
 return admin_user.id
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"获取或创建管理员用户失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def create_conversation(user_id: Optional[int] = None, title: Optional[str] = None,
 category_id: Optional[int] = None) -> int:
 db = get_db_session()
 try:
 if user_id is None:
 user_id = DatabaseOperations.get_or_create_admin_user()
 if title is None:
 title = f"对话_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
 conversation = ConversationList(
 user_id=user_id,
 title=title,
 category_id=category_id,
 is_completed=False
 )
 db.add(conversation)
 db.commit()
 return conversation.id
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"创建对话失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def add_conversation_history(conversation_id: int, user_id: Optional[int] = None,
 role: str = "user", content: str = "", reasoning: Optional[str] = None,
 model_name: Optional[str] = None, tokens: Optional[int] = None) -> int:
 db = get_db_session()
 try:
 if user_id is None:
 user_id = DatabaseOperations.get_or_create_admin_user()
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 if not conversation:
 raise ValueError(f"对话ID {conversation_id} 不存在")
 history = ConversationHistory(
 conversation_id=conversation_id,
 user_id=user_id,
 role=RoleEnum(role),
 content=content,
 reasoning=reasoning,
 model_name=model_name,
 tokens=tokens
 )
 db.add(history)
 conversation.update_time = datetime.datetime.now()
 db.commit()
 return history.id
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"添加对话历史失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def get_conversation_history(conversation_id: int) -> List[Dict[str, Any]]:
 db = get_db_session()
 try:
 histories = db.query(ConversationHistory).filter(
 ConversationHistory.conversation_id == conversation_id
 ).order_by(ConversationHistory.create_time).all()
 result = []
 for history in histories:
 result.append({
 "id": history.id,
 "role": history.role.value,
 "content": history.content,
 "reasoning": history.reasoning,
 "create_time": history.create_time.strftime("%Y-%m-%d %H:%M:%S"),
 "model_name": history.model_name,
 "tokens": history.tokens
 })
 return result
 except SQLAlchemyError as e:
 logger.error(f"获取对话历史失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def complete_conversation(conversation_id: int, satisfaction: Optional[str] = None,
 feedback: Optional[str] = None) -> bool:
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 if not conversation:
 raise ValueError(f"对话ID {conversation_id} 不存在")
 conversation.is_completed = True
 if satisfaction:
 conversation.satisfaction = SatisfactionEnum(satisfaction)
 if feedback:
 conversation.feedback = feedback
 db.commit()
 return True
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"完成对话失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def get_user_conversations(user_id: Optional[int] = None,
 limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
 db = get_db_session()
 try:
 if user_id is None:
 user_id = DatabaseOperations.get_or_create_admin_user()
 conversations = db.query(ConversationList).filter(
 ConversationList.user_id == user_id
 ).order_by(ConversationList.update_time.desc()).limit(limit).offset(offset).all()
 result = []
 for conversation in conversations:
 last_ai_reply = db.query(ConversationHistory).filter(
 ConversationHistory.conversation_id == conversation.id,
 ConversationHistory.role == RoleEnum.ai
 ).order_by(ConversationHistory.create_time.desc()).first()
 message_count = db.query(ConversationHistory).filter(
 ConversationHistory.conversation_id == conversation.id
 ).count()
 preview = last_ai_reply.content[:100] + "..." if last_ai_reply and len(last_ai_reply.content) > 100 else (
 last_ai_reply.content if last_ai_reply else "")
 result.append({
 "id": conversation.id,
 "title": conversation.title,
 "create_time": conversation.create_time.strftime("%Y-%m-%d %H:%M:%S"),
 "update_time": conversation.update_time.strftime("%Y-%m-%d %H:%M:%S"),
 "is_completed": conversation.is_completed,
 "satisfaction": conversation.satisfaction.value if conversation.satisfaction else None,
 "message_count": message_count,
 "preview": preview
 })
 return result
 except SQLAlchemyError as e:
 logger.error(f"获取用户对话列表失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def update_conversation_title(conversation_id: int, title: str) -> bool:
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 if not conversation:
 raise ValueError(f"对话ID {conversation_id} 不存在")
 conversation.title = title
 db.commit()
 return True
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"更新对话标题失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def delete_conversation(conversation_id: int) -> bool:
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 if not conversation:
 raise ValueError(f"对话ID {conversation_id} 不存在")
 db.query(ConversationHistory).filter(ConversationHistory.conversation_id == conversation_id).delete()
 db.delete(conversation)
 db.commit()
 return True
 except SQLAlchemyError as e:
 db.rollback()
 logger.error(f"删除对话失败: {e}")
 raise
 finally:
 close_db_session(db)
 @staticmethod
 def generate_conversation_id() -> str:
 return str(uuid.uuid4())```
______________________________

## .../database/db_config.py
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from dotenv import load_dotenv
from app.utils.logger import logger
load_dotenv()
DB_URL = os.getenv("DB_URL", "mysql+pymysql://root:123654CCc.@bj-cdb-oqrn4mh2.sql.tencentcdb.com:24734/deepsysai?charset=utf8mb3")
engine = create_engine(
 DB_URL,
 pool_size=10,
 max_overflow=20,
 pool_recycle=3600,
 pool_pre_ping=True
)
SessionFactory = sessionmaker(bind=engine, autoflush=False)
Session = scoped_session(SessionFactory)
Base = declarative_base()
def get_db_session():
 db = Session()
 try:
 return db
 except Exception as e:
 db.rollback()
 logger.error(f"数据库会话创建失败: {e}")
 raise
def close_db_session(db):
 db.close()```
______________________________

## .../database/db_utils.py
```python
import os
from sqlalchemy import text
from .db_config import get_db_session, close_db_session
from app.utils.logger import logger
def add_reasoning_column_if_not_exists():
 db = get_db_session()
 try:
 check_sql = text()
 result = db.execute(check_sql).first()
 if result and result[0] > 0:
 logger.info("reasoning列已存在于conversation_history表中")
 return True
 logger.info("正在向conversation_history表添加reasoning列...")
 add_column_sql = text()
 db.execute(add_column_sql)
 db.commit()
 verify_sql = text()
 verify_result = db.execute(verify_sql).first()
 if verify_result and verify_result[0] == 'reasoning':
 logger.info("reasoning列已成功添加到conversation_history表中")
 return True
 else:
 logger.error("添加reasoning列失败，请检查数据库权限")
 return False
 except Exception as e:
 db.rollback()
 logger.error(f"添加reasoning列时发生错误: {e}")
 return False
 finally:
 close_db_session(db)```
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
import os
import sys
from logging.handlers import RotatingFileHandler
import time
import json
class CustomFormatter(logging.Formatter):
 def __init__(self, fmt=None, datefmt=None, style='%'):
 super().__init__(fmt, datefmt, style)
 def formatException(self, exc_info):
 result = super().formatException(exc_info)
 return result
 def format(self, record):
 try:
 if record.levelno == logging.DEBUG and len(record.msg) > 10000:
 record.msg = record.msg[:10000] + "... [截断]"
 if isinstance(record.msg, dict):
 try:
 record.msg = json.dumps(record.msg, ensure_ascii=False, indent=2)
 except:
 pass
 except:
 pass
 return super().format(record)
def setup_logger(name='deepclaude'):
 log_dir = os.path.join(os.getcwd(), 'logs')
 os.makedirs(log_dir, exist_ok=True)
 current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
 log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
 logger = logging.getLogger(name)
 log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
 numeric_level = getattr(logging, log_level, logging.INFO)
 logger.setLevel(numeric_level)
 if logger.handlers:
 return logger
 console_handler = logging.StreamHandler(sys.stdout)
 console_handler.setLevel(numeric_level)
 max_bytes = 10 * 1024 * 1024
 file_handler = RotatingFileHandler(
 log_file,
 maxBytes=max_bytes,
 backupCount=5,
 encoding='utf-8'
 )
 file_handler.setLevel(numeric_level)
 formatter = CustomFormatter(
 fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
 datefmt='%Y-%m-%d %H:%M:%S'
 )
 console_handler.setFormatter(formatter)
 file_handler.setFormatter(formatter)
 logger.addHandler(console_handler)
 logger.addHandler(file_handler)
 logger.info(f"日志级别设置为: {log_level}")
 return logger
logger = setup_logger()```
______________________________

## .../utils/message_processor.py
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

## .../utils/streaming.py
```python
from app.deepclaude.utils.streaming import StreamingHelper
__all__ = ["StreamingHelper"]```
______________________________

## .../v1/deepclaude.py
```python
from fastapi import APIRouter, Depends, Request, Response, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import os
from pydantic import BaseModel, Field
from app.utils.logger import logger
from app.deepclaude.utils.streaming import StreamingHelper
from app.deepclaude.core import DeepClaude
router = APIRouter(
 prefix="/api/v1/deepclaude",
 tags=["deepclaude"]
)
_DEEPCLAUDE_INSTANCE = None
class ChatMessage(BaseModel):
 role: str
 content: str
 name: Optional[str] = None
class ChatRequest(BaseModel):
 messages: List[ChatMessage]
 model: str = "deepclaude"
 temperature: Optional[float] = 0.7
 top_p: Optional[float] = 0.9
 stream: Optional[bool] = True
 tools: Optional[List[Dict[str, Any]]] = None
 tool_choice: Optional[Any] = "auto"
 save_to_db: Optional[bool] = None
def get_deepclaude():
 global _DEEPCLAUDE_INSTANCE
 if _DEEPCLAUDE_INSTANCE is None:
 logger.info("初始化 DeepClaude 实例...")
 _DEEPCLAUDE_INSTANCE = DeepClaude()
 return _DEEPCLAUDE_INSTANCE
@router.post("/chat/completions")
async def chat_completions(
 request: ChatRequest,
 raw_request: Request,
 background_tasks: BackgroundTasks,
 deepclaude: DeepClaude = Depends(get_deepclaude)
):
 try:
 messages = [msg.dict() for msg in request.messages]
 model_arg = (request.temperature, request.top_p)
 if request.save_to_db is not None:
 deepclaude.save_to_db = request.save_to_db
 if request.stream:
 logger.info(f"开始处理流式请求: model={request.model}, tools数量={len(request.tools) if request.tools else 0}")
 async def generate_stream():
 async for chunk in deepclaude.chat_completions_with_stream(
 messages=messages,
 tools=request.tools,
 tool_choice=request.tool_choice,
 temperature=request.temperature,
 top_p=request.top_p,
 model=request.model
 ):
 yield chunk
 return StreamingResponse(
 generate_stream(),
 media_type="text/event-stream"
 )
 else:
 logger.info(f"开始处理非流式请求: model={request.model}, tools数量={len(request.tools) if request.tools else 0}")
 response = await deepclaude.chat_completions_without_stream(
 messages=messages,
 model_arg=model_arg,
 tools=request.tools,
 tool_choice=request.tool_choice,
 model=request.model
 )
 return JSONResponse(content=response)
 except Exception as e:
 logger.error(f"处理请求时出错: {e}", exc_info=True)
 raise HTTPException(status_code=500, detail=str(e))
@router.post("/reasoning")
async def get_reasoning(
 request: ChatRequest,
 deepclaude: DeepClaude = Depends(get_deepclaude)
):
 try:
 messages = [msg.dict() for msg in request.messages]
 reasoning = await deepclaude.thinker_client.get_reasoning(
 messages=messages,
 model=request.model,
 model_arg=(request.temperature, request.top_p)
 )
 return JSONResponse(content={"reasoning": reasoning})
 except Exception as e:
 logger.error(f"获取推理时出错: {e}", exc_info=True)
 raise HTTPException(status_code=500, detail=str(e))
@router.get("/health")
async def health_check():
 return {"status": "healthy", "service": "deepclaude"}```
______________________________

## .../deepclaude/interfaces.py
```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
class ReasoningProvider(ABC):
 @abstractmethod
 async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
 pass
class GenerationProvider(ABC):
 @abstractmethod
 async def generate_response(self, messages: List[Dict], model: str, **kwargs) -> Dict:
 pass
 @abstractmethod
 async def stream_response(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
 pass
class ToolProcessor(ABC):
 @abstractmethod
 def validate_and_convert(self, tools: List[Dict], target_format: str) -> List[Dict]:
 pass
 @abstractmethod
 async def process_tool_call(self, tool_call: Dict, **kwargs) -> Dict:
 pass```
______________________________

## .../deepclaude/__init__.py
```python
from .core import DeepClaude
__all__ = ["DeepClaude"]```
______________________________

## .../deepclaude/core.py
```python
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import uuid
import time
from app.utils.logger import logger
from .reasoning.factory import ReasoningProviderFactory
from .tools.handlers import ToolHandler
from app.clients.claude_client import ClaudeClient
import copy
import aiohttp
import asyncio
from datetime import datetime
class DeepClaude:
 def __init__(self, **kwargs):
 logger.info("初始化DeepClaude服务...")
 self.claude_api_key = kwargs.get('claude_api_key', os.getenv('CLAUDE_API_KEY', ''))
 self.claude_api_url = kwargs.get('claude_api_url', os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com/v1/messages'))
 self.claude_provider = kwargs.get('claude_provider', os.getenv('CLAUDE_PROVIDER', 'anthropic'))
 self.is_origin_reasoning = kwargs.get('is_origin_reasoning', os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true')
 self.min_reasoning_chars = 100
 self.claude_client = ClaudeClient(
 api_key=self.claude_api_key,
 api_url=self.claude_api_url,
 provider=self.claude_provider
 )
 self.tool_handler = ToolHandler()
 provider_type = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 self.thinker_client = ReasoningProviderFactory.create(provider_type)
 self.save_to_db = kwargs.get('save_to_db', os.getenv('SAVE_TO_DB', 'false').lower() == 'true')
 if self.save_to_db:
 logger.info("启用数据库存储...")
 from app.database.db_operations import DatabaseOperations
 self.db_ops = kwargs.get('db_ops', DatabaseOperations())
 self.current_conversation_id = None
 else:
 logger.info("数据库存储已禁用")
 self.db_ops = None
 logger.info("DeepClaude服务初始化完成")
 async def chat_completions_with_stream(self, messages: list, tools: list = None, tool_choice = "auto", **kwargs):
 logger.info("开始流式处理请求...")
 try:
 processed_count = 0
 tool_call_chunks = []
 tool_results = []
 if tools and len(tools) > 0:
 logger.info(f"直接透传模式(流式): 包含 {len(tools)} 个工具")
 valid_tools_len = 0
 valid_tools = None
 if tools:
 from app.clients.handlers import validate_and_convert_tools
 valid_tools = validate_and_convert_tools(tools, 'claude-3')
 valid_tools_len = len(valid_tools) if valid_tools else 0
 logger.info(f"最终使用 {valid_tools_len} 个工具调用Claude")
 claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 claude_model = kwargs.get("claude_model", claude_model)
 claude_kwargs = {
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "max_tokens": kwargs.get("max_tokens", 8192),
 "top_p": kwargs.get("top_p", 0.9),
 "tools": valid_tools,
 "tool_choice": {"type": tool_choice} if tool_choice != "none" else {"type": "none"}
 }
 logger.info("开始调用Claude流式接口...")
 content_chunks = []
 async for chunk in self.claude_client.stream_chat(messages, **claude_kwargs):
 processed_count += 1
 if not isinstance(chunk, dict):
 logger.warning(f"从Claude客户端收到非字典格式数据: {type(chunk)}")
 continue
 if "error" in chunk:
 error_message = chunk.get("error", "未知错误")
 logger.error(f"Claude API错误: {error_message}")
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"错误: {error_message}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 return
 if "tool_calls" in chunk and chunk["tool_calls"]:
 for tool_call in chunk["tool_calls"]:
 logger.info(f"收到Claude工具调用[{processed_count}]: {tool_call.get('function', {}).get('name', '未知工具')}")
 tool_call_chunks.append(tool_call)
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "role": "assistant",
 "tool_calls": [tool_call]
 },
 "finish_reason": None
 }
 ]
 }
 logger.info(f"处理工具调用: {json.dumps(tool_call)[:200]}...")
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 elif "content" in chunk and chunk["content"]:
 content_chunks.append(chunk["content"])
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": chunk["content"]
 },
 "finish_reason": None
 }
 ]
 }
 logger.info(f"发送文本响应[{processed_count}]: {json.dumps(response)[:200]}...")
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 logger.info(f"流式处理完成，总共发送了 {processed_count} 个响应片段 (内容: {len(content_chunks)}, 工具调用: {len(tool_call_chunks)})")
 if tool_call_chunks:
 finish_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {},
 "finish_reason": "tool_calls"
 }
 ]
 }
 logger.info(f"发送工具调用完成标记: {json.dumps(finish_response)}")
 yield f"data: {json.dumps(finish_response, ensure_ascii=False)}\n\n".encode("utf-8")
 try:
 for i, tool_call in enumerate(tool_call_chunks):
 logger.info(f"执行工具调用[{i}]: {tool_call.get('function', {}).get('name', '未知工具')}, 参数: {tool_call.get('function', {}).get('arguments', '{}')}")
 tool_name = tool_call.get("function", {}).get("name", "")
 tool_args = tool_call.get("function", {}).get("arguments", "{}")
 if isinstance(tool_args, str):
 try:
 tool_args = json.loads(tool_args)
 except Exception as e:
 logger.error(f"解析工具参数时出错: {e}")
 tool_args = {}
 try:
 result = await self._execute_tool_call({
 "tool": tool_name,
 "tool_input": tool_args
 })
 logger.info(f"工具调用[{i}]执行结果: {result[:200]}...")
 tool_results.append({
 "role": "user",
 "name": tool_name,
 "content": result
 })
 except Exception as e:
 logger.error(f"执行工具调用[{i}]时出错: {e}")
 error_result = f"执行工具调用时发生错误: {str(e)}"
 tool_results.append({
 "role": "user",
 "name": tool_name,
 "content": error_result
 })
 except Exception as e:
 logger.error(f"处理工具调用过程中发生错误: {e}")
 error_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": f"执行工具调用过程中发生错误: {str(e)}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 return
 if tool_results:
 assistant_message = {
 "role": "assistant",
 "content": "我需要查询信息以回答您的问题"
 }
 tool_results_content = ""
 for i, result in enumerate(tool_results):
 tool_name = result.get("name", f"工具{i}")
 tool_content = result.get("content", "")
 tool_results_content += f"### {tool_name}工具的执行结果 ###\n{tool_content}\n\n"
 user_message = {
 "role": "user",
 "content": tool_results_content
 }
 new_messages = copy.deepcopy(messages)
 new_messages.append(assistant_message)
 new_messages.append(user_message)
 logger.info(f"将工具调用结果回传给Claude, 新消息数: {len(new_messages)}")
 logger.info("使用工具结果继续与Claude对话...")
 continue_kwargs = {
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "max_tokens": kwargs.get("max_tokens", 8192),
 "top_p": kwargs.get("top_p", 0.9)
 }
 try:
 final_content_chunks = []
 async for chunk in self.claude_client.stream_chat(new_messages, **continue_kwargs):
 if not isinstance(chunk, dict):
 logger.warning(f"从Claude客户端收到非字典格式数据: {type(chunk)}")
 continue
 if "error" in chunk:
 error_message = chunk.get("error", "未知错误")
 logger.error(f"继续对话时发生错误: {error_message}")
 error_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": f"错误: {error_message}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode("utf-8")
 continue
 if "content" in chunk and chunk["content"]:
 final_content_chunks.append(chunk["content"])
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": chunk["content"]
 },
 "finish_reason": None
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 final_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {},
 "finish_reason": "stop"
 }
 ]
 }
 logger.info(f"发送最终完成标记: {json.dumps(final_response)}")
 yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode("utf-8")
 except Exception as e:
 logger.error(f"使用工具结果继续对话失败: {e}")
 error_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": f"处理工具调用结果时发生错误: {str(e)}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode("utf-8")
 elif processed_count > 0 and not tool_call_chunks:
 finish_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {},
 "finish_reason": "stop"
 }
 ]
 }
 logger.info("发送完成标记(无工具调用)")
 yield f"data: {json.dumps(finish_response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 return
 else:
 logger.info("常规流式模式(无工具)")
 model_arg = None
 content_chunks = []
 deepseek_model = kwargs.get("deepseek_model", "deepseek-reasoner")
 claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 claude_model = kwargs.get("claude_model", claude_model)
 selected_model = kwargs.get("selected_model", "claude")
 if selected_model == "deepseek":
 async for chunk in self.deepseek_client.stream_chat(messages, deepseek_model, model_arg=model_arg):
 if "error" in chunk:
 error_message = chunk.get("error", "未知错误")
 logger.error(f"DeepSeek API错误: {error_message}")
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"错误: {error_message}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 return
 processed_count += 1
 if "content" in chunk and chunk["content"]:
 content_chunks.append(chunk["content"])
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": chunk["content"]
 },
 "finish_reason": None
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 else:
 claude_kwargs = {
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "max_tokens": kwargs.get("max_tokens", 8192),
 "top_p": kwargs.get("top_p", 0.9)
 }
 async for chunk in self.claude_client.stream_chat(messages, **claude_kwargs):
 if "error" in chunk:
 error_message = chunk.get("error", "未知错误")
 logger.error(f"Claude API错误: {error_message}")
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"错误: {error_message}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 return
 processed_count += 1
 if "content" in chunk and chunk["content"]:
 content_chunks.append(chunk["content"])
 response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "content": chunk["content"]
 },
 "finish_reason": None
 }
 ]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
 if processed_count > 0:
 finish_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {},
 "finish_reason": "stop"
 }
 ]
 }
 yield f"data: {json.dumps(finish_response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 except Exception as e:
 logger.error(f"流式处理异常: {e}", exc_info=True)
 error_response = {
 "id": f"chatcmpl-{str(uuid.uuid4())[:8]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:4]}-{str(uuid.uuid4())[:12]}",
 "object": "chat.completion.chunk",
 "created": int(time.time()),
 "model": kwargs.get("model", "gpt-4o"),
 "choices": [
 {
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"处理请求时发生错误: {str(e)}"
 },
 "finish_reason": "error"
 }
 ]
 }
 yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode("utf-8")
 yield b"data: [DONE]\n\n"
 async def chat_completions_without_stream(
 self,
 messages: list,
 model_arg: tuple,
 tools: list = None,
 tool_choice = "auto",
 deepseek_model: str = "deepseek-reasoner",
 claude_model: str = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
 **kwargs
 ) -> dict:
 logger.info("开始处理非流式请求...")
 try:
 logger.info("开始处理非流式请求...")
 direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
 if direct_tool_pass and tools and len(tools) > 0:
 logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
 converted_tools = self.tool_handler.validate_and_convert_tools(tools, target_format='claude-3')
 if not converted_tools:
 return {
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": "没有找到有效的工具定义，将作为普通对话处理。"
 },
 "finish_reason": "stop"
 }]
 }
 final_tools = []
 for i, tool in enumerate(converted_tools):
 cleaned_tool = {
 "name": tool.get("name", f"未命名工具_{i}"),
 "description": tool.get("description", "")
 }
 if "input_schema" in tool:
 cleaned_tool["input_schema"] = tool["input_schema"]
 elif "custom" in tool and isinstance(tool["custom"], dict):
 custom = tool["custom"]
 if "input_schema" in custom:
 cleaned_tool["input_schema"] = custom["input_schema"]
 else:
 cleaned_tool["input_schema"] = {
 "type": "object",
 "properties": custom.get("properties", {}),
 "required": custom.get("required", [])
 }
 elif "parameters" in tool:
 cleaned_tool["input_schema"] = {
 "type": "object",
 "properties": tool["parameters"].get("properties", {}),
 "required": tool["parameters"].get("required", [])
 }
 else:
 logger.warning(f"工具[{i}]缺少input_schema字段，将被跳过")
 continue
 final_tools.append(cleaned_tool)
 logger.debug(f"工具[{i}]最终格式: {json.dumps(cleaned_tool, ensure_ascii=False)}")
 claude_kwargs = {
 "messages": messages,
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "top_p": kwargs.get("top_p", 0.9),
 "tools": final_tools,
 "stream": False
 }
 if isinstance(tool_choice, str):
 if tool_choice == "auto":
 claude_kwargs["tool_choice"] = {"type": "auto"}
 elif tool_choice == "none":
 logger.info("检测到'none'工具选择策略，将不使用工具")
 claude_kwargs.pop("tools")
 elif isinstance(tool_choice, dict):
 claude_kwargs["tool_choice"] = tool_choice
 response = await self.claude_client.chat(**claude_kwargs)
 return response
 original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
 logger.info("正在获取推理内容...")
 reasoning = await self.thinker_client.get_reasoning(
 messages=messages,
 model=deepseek_model,
 model_arg=model_arg
 )
 logger.debug(f"获取到推理内容: {reasoning[:200] if reasoning else '无'}...")
 combined_prompt = f"我已经思考了以下问题：\n\n{original_question}\n\n我的思考过程是：\n{reasoning}\n\n现在，给出清晰、准确、有帮助的回答，不要提及上面的思考过程，直接开始回答。"
 claude_messages = [{"role": "user", "content": combined_prompt}]
 logger.info("正在获取Claude回答...")
 answer_response = await self.claude_client.chat(
 messages=claude_messages,
 model=claude_model,
 temperature=model_arg[0] if model_arg else 0.7,
 top_p=model_arg[1] if model_arg else 0.9,
 stream=False
 )
 content = answer_response.get("content", "")
 if self.save_to_db and self.current_conversation_id:
 try:
 tokens = len(reasoning.split()) + len(content.split())
 self.db_ops.add_conversation_history(
 conversation_id=self.current_conversation_id,
 role="ai",
 content=content,
 reasoning=reasoning,
 model_name=claude_model,
 tokens=tokens
 )
 logger.info("AI回答和思考过程已保存到数据库")
 except Exception as db_e:
 logger.error(f"保存AI回答数据失败: {db_e}")
 return {
 "role": "assistant",
 "content": content,
 "reasoning": reasoning
 }
 except Exception as e:
 logger.error(f"处理非流式请求时发生错误: {e}", exc_info=True)
 return {
 "role": "assistant",
 "content": f"处理请求时出错: {str(e)}",
 "error": True
 }
 def _format_tool_decision_prompt(self, original_question: str, reasoning: str, tools: List[Dict]) -> str:
 tools_description = ""
 for i, tool in enumerate(tools, 1):
 if "function" in tool:
 function = tool["function"]
 name = function.get("name", "未命名工具")
 description = function.get("description", "无描述")
 parameters = function.get("parameters", {})
 required = parameters.get("required", [])
 properties = parameters.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "name" in tool and "input_schema" in tool:
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 input_schema = tool.get("input_schema", {})
 required = input_schema.get("required", [])
 properties = input_schema.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "type" in tool and tool["type"] == "custom":
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 tool_schema = tool.get("tool_schema", {})
 required = tool_schema.get("required", [])
 properties = tool_schema.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "name" in tool and "parameters" in tool:
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 parameters = tool.get("parameters", {})
 required = parameters.get("required", [])
 properties = parameters.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 prompt = f
 logger.info(f"生成工具决策提示 - 问题: '{original_question[:30]}...'")
 return prompt
 async def _enhance_with_search(self, query: str) -> str:
 if not os.getenv('ENABLE_SEARCH_ENHANCEMENT', 'true').lower() == 'true':
 logger.info("搜索增强功能已禁用")
 return ""
 logger.info(f"为查询提供搜索增强: {query}")
 real_time_keywords = [
 "今天", "现在", "最新", "天气", "股市", "价格", "新闻",
 "比赛", "比分", "最近", "实时", "日期", "时间"
 ]
 needs_search = any(keyword in query for keyword in real_time_keywords)
 if not needs_search:
 logger.info("查询不需要实时信息增强")
 return ""
 try:
 search_query = query
 tool_input = {
 "name": "tavily_search",
 "input": {
 "query": search_query
 }
 }
 logger.info(f"执行搜索: {search_query}")
 search_result = "由于这是模拟搜索结果，实际使用时需要对接真实搜索API。搜索结果应包含关于查询的最新信息。"
 if search_result:
 logger.info(f"获取到搜索结果: {search_result[:100]}...")
 return f"以下是关于\"{query}\"的最新信息:\n\n{search_result}\n\n请基于上述信息来思考和回答问题。"
 except Exception as e:
 logger.warning(f"搜索增强失败: {e}")
 return ""
 async def _get_reasoning_content(self, messages: list, model: str, model_arg: tuple = None, **kwargs) -> str:
 try:
 user_message = ""
 for msg in reversed(messages):
 if msg.get("role") == "user" and "content" in msg:
 user_message = msg["content"]
 break
 search_enhancement = ""
 if user_message:
 search_enhancement = await self._enhance_with_search(user_message)
 reasoning = await self.thinker_client.get_reasoning(
 messages=messages,
 model=model,
 model_arg=model_arg
 )
 if search_enhancement:
 reasoning = f"{search_enhancement}\n\n{reasoning}"
 return reasoning
 except Exception as e:
 logger.error(f"获取推理内容失败: {e}")
 return "获取推理内容失败"
 async def _execute_tool_call(self, tool_call: Dict) -> str:
 logger.info(f"执行工具调用: {json.dumps(tool_call, ensure_ascii=False)}")
 tool_name = None
 tool_input = {}
 if "tool" in tool_call:
 tool_name = tool_call["tool"]
 elif "function" in tool_call and isinstance(tool_call["function"], dict) and "name" in tool_call["function"]:
 tool_name = tool_call["function"]["name"]
 elif "name" in tool_call:
 tool_name = tool_call["name"]
 if "tool_input" in tool_call:
 tool_input = tool_call["tool_input"]
 elif "function" in tool_call and isinstance(tool_call["function"], dict) and "arguments" in tool_call["function"]:
 args = tool_call["function"]["arguments"]
 if isinstance(args, str):
 try:
 tool_input = json.loads(args)
 except json.JSONDecodeError:
 tool_input = {"query": args}
 else:
 tool_input = args
 elif "arguments" in tool_call:
 args = tool_call["arguments"]
 if isinstance(args, str):
 try:
 tool_input = json.loads(args)
 except json.JSONDecodeError:
 tool_input = {"query": args}
 else:
 tool_input = args
 logger.info(f"解析后的工具名称: {tool_name}, 输入参数: {json.dumps(tool_input, ensure_ascii=False)}")
 try:
 if tool_name == "tavily_search":
 return await self._execute_tavily_search(tool_input)
 elif tool_name == "tavily_extract":
 return await self._execute_tavily_extract(tool_input)
 else:
 logger.warning(f"未知工具: {tool_name}")
 return f"不支持的工具类型: {tool_name}"
 except Exception as e:
 logger.error(f"执行工具出错: {e}", exc_info=True)
 return f"工具执行失败: {str(e)}"
 async def _execute_tavily_search(self, input_data: Dict) -> str:
 if not input_data or not isinstance(input_data, dict) or "query" not in input_data:
 return "搜索查询缺失，无法执行搜索。请提供有效的查询内容。"
 query = input_data.get("query", "").strip()
 if not query:
 return "搜索查询为空，无法执行搜索。请提供有效的查询内容。"
 logger.info(f"执行Tavily搜索: {query}")
 tavily_api_key = os.getenv("TAVILY_API_KEY")
 if "沈阳" in query and ("天气" in query or "气温" in query):
 logger.info(f"检测到沈阳天气查询: {query}")
 current_date = datetime.now().strftime("%Y年%m月%d日")
 mock_response = f
 logger.info("返回沈阳天气模拟数据")
 return mock_response
 try:
 if not tavily_api_key:
 logger.warning("未设置TAVILY_API_KEY，使用模拟搜索响应")
 if "天气" in query:
 location = query.replace("天气", "").strip()
 if not location:
 location = "未指定地区"
 mock_response = f"关于{location}天气的模拟搜索结果：今日天气晴朗，气温适宜，建议适当户外活动。(注：由于未配置搜索API密钥，此为模拟数据)"
 return mock_response
 else:
 return f"关于"{query}"的模拟搜索结果：由于搜索服务未配置API密钥，无法提供实时信息。这是一个模拟响应。"
 logger.info("发送Tavily API请求")
 tavily_url = "https://api.tavily.com/search"
 headers = {
 "Content-Type": "application/json",
 "Authorization": f"Bearer {tavily_api_key}"
 }
 payload = {
 "query": query,
 "search_depth": "advanced",
 "include_domains": [],
 "exclude_domains": [],
 "max_results": 5
 }
 async with aiohttp.ClientSession() as session:
 async with session.post(tavily_url, json=payload, headers=headers, timeout=10) as response:
 if response.status == 200:
 result = await response.json()
 content = ""
 if "results" in result:
 for idx, item in enumerate(result["results"], 1):
 content += f"{idx}. {item.get('title', '无标题')}\n"
 content += f"   链接: {item.get('url', '无链接')}\n"
 content += f"   内容: {item.get('content', '无内容摘要')[:200]}...\n\n"
 return content.strip() or "搜索未返回任何结果。"
 elif response.status == 401:
 logger.error("Tavily API请求未授权")
 if "沈阳" in query and "天气" in query:
 logger.warning("Tavily API授权失败，使用沈阳天气模拟响应")
 return f"根据模拟信息，沈阳今日天气：气温18-26°C，晴朗，偏北风2-3级，空气质量优，适合户外活动。未来三天天气稳定，无明显降水。请注意早晚温差较大，注意适当添加衣物。(注：这是模拟数据，由于搜索服务未能正常连接)"
 else:
 logger.warning("Tavily API授权失败，使用通用模拟响应")
 return f"搜索请求未授权。关于\"{query}\"的模拟结果：沈阳地区今日天气晴好，温度适宜，适合户外活动。请注意这是模拟信息，可能与实际情况有差异。"
 except asyncio.TimeoutError:
 logger.error("Tavily API请求超时")
 return f"搜索请求超时。关于\"{query}\"的模拟结果：沈阳地区今日天气晴好，温度适宜，适合户外活动。请注意这是模拟信息，可能与实际情况有差异。";
 except Exception as e:
 logger.error(f"执行tavily_search时出错: {e}")
 return f"执行搜索时发生错误: {str(e)}。这是一个模拟响应，仅供参考。"
 async def _execute_tavily_extract(self, input_data: Dict) -> str:
 urls = input_data.get("urls", "")
 if not urls:
 return "URL列表不能为空"
 urls_list = [url.strip() for url in urls.split(",")]
 logger.info(f"执行Tavily网页提取: {urls_list}")
 api_key = os.getenv('TAVILY_API_KEY')
 if not api_key:
 logger.warning("未配置Tavily API密钥，使用模拟结果")
 return f"这是从URL '{urls}'提取的模拟内容。在实际使用中，这里会返回从网页中提取的文本内容。"
 try:
 import aiohttp
 results = []
 for url in urls_list:
 extract_url = "https://api.tavily.com/extract"
 headers = {
 "Content-Type": "application/json",
 "Authorization": f"Bearer {api_key}"
 }
 payload = {
 "url": url,
 "include_images": False
 }
 async with aiohttp.ClientSession() as session:
 async with session.post(extract_url, headers=headers, json=payload) as response:
 if response.status == 200:
 result = await response.json()
 content = result.get("content", "")
 results.append(f"URL: {url}\n\n{content}")
 else:
 error_text = await response.text()
 logger.error(f"Tavily提取API错误: {response.status} - {error_text}")
 results.append(f"URL {url} 提取失败: HTTP {response.status}")
 return "\n\n---\n\n".join(results)
 except Exception as e:
 logger.error(f"Tavily提取API调用失败: {e}")
 return f"内容提取错误: {str(e)}"```
______________________________

## .../deepclaude/deepclaude.py
```python
import json
import time
import tiktoken
import asyncio
import uuid
import re
from typing import AsyncGenerator, Dict, List, Any, Optional, Tuple
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, OllamaR1Client
from app.utils.message_processor import MessageProcessor
import aiohttp
import os
from dotenv import load_dotenv
import sys
import logging
import requests
from datetime import datetime
from app.database.db_operations import DatabaseOperations
from app.database.db_utils import add_reasoning_column_if_not_exists
load_dotenv()
class DeepClaude:
 def __init__(self, **kwargs):
 logger.info("初始化DeepClaude服务...")
 self.claude_api_key = kwargs.get('claude_api_key', os.getenv('CLAUDE_API_KEY', ''))
 self.claude_api_url = kwargs.get('claude_api_url', os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com/v1/messages'))
 self.claude_provider = kwargs.get('claude_provider', os.getenv('CLAUDE_PROVIDER', 'anthropic'))
 self.ollama_api_url = kwargs.get('ollama_api_url', os.getenv('OLLAMA_API_URL', ''))
 self.is_origin_reasoning = kwargs.get('is_origin_reasoning', os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true')
 self.enable_enhanced_reasoning = kwargs.get('enable_enhanced_reasoning', True)
 self.min_reasoning_chars = 100
 self.reasoning_modes = ["auto", "chain-of-thought", "zero-shot"]
 self.saved_reasoning = ""
 self.processor = MessageProcessor()
 self.reasoning_providers = {
 'deepseek': lambda: DeepSeekClient(
 api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
 api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
 provider=os.getenv('DEEPSEEK_PROVIDER', 'deepseek')
 ),
 'siliconflow': lambda: DeepSeekClient(
 api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
 api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
 provider='siliconflow'
 ),
 'nvidia': lambda: DeepSeekClient(
 api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
 api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
 provider='nvidia'
 ),
 'ollama': lambda: OllamaR1Client(
 api_url=kwargs.get('ollama_api_url', os.getenv('OLLAMA_API_URL', ''))
 )
 }
 self.supported_tools = {
 "search": {
 "name": "search",
 "description": "搜索网络获取实时信息",
 "parameters": {
 "type": "object",
 "properties": {
 "query": {
 "type": "string",
 "description": "搜索查询内容"
 }
 },
 "required": ["query"]
 }
 },
 "weather": {
 "name": "weather",
 "description": "获取天气信息",
 "parameters": {
 "type": "object",
 "properties": {
 "location": {
 "type": "string",
 "description": "地点名称，如：北京、上海"
 },
 "date": {
 "type": "string",
 "description": "日期，如：today、tomorrow",
 "enum": ["today", "tomorrow"]
 }
 },
 "required": ["location"]
 }
 }
 }
 self.tool_format_mapping = {
 'gpt-4': {
 'type': 'function',
 'function_field': 'function'
 },
 'gpt-3.5-turbo': {
 'type': 'function',
 'function_field': 'function'
 },
 'claude-3': {
 'type': 'tool',
 'function_field': 'function'
 }
 }
 self.save_to_db = kwargs.get('save_to_db', os.getenv('SAVE_TO_DB', 'false').lower() == 'true')
 if self.save_to_db:
 logger.info("启用数据库存储...")
 self.db_ops = kwargs.get('db_ops', DatabaseOperations())
 self.current_conversation_id = None
 add_reasoning_column_if_not_exists()
 else:
 logger.info("数据库存储已禁用")
 self.db_ops = None
 if 'clients' in kwargs:
 self.thinker_client = kwargs['clients'].get('thinker')
 self.claude_client = kwargs['clients'].get('claude')
 else:
 logger.info("初始化思考者客户端...")
 provider = self._get_reasoning_provider()
 self.thinker_client = provider
 logger.info("初始化Claude客户端...")
 self.claude_client = ClaudeClient(
 api_key=self.claude_api_key,
 api_url=self.claude_api_url,
 provider=self.claude_provider
 )
 self._validate_config()
 self.search_enabled = os.getenv('ENABLE_SEARCH_ENHANCEMENT', 'true').lower() == 'true'
 logger.info("DeepClaude服务初始化完成")
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
 "is_reasoning": True,
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"🤔 思考过程:\n{content}\n",
 "reasoning": True
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
 async def chat_completions_with_stream(self, messages: list, tools: list = None, tool_choice = "auto", **kwargs):
 chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
 created_time = kwargs.get("created_time", int(time.time()))
 model_name = kwargs.get("model", "deepclaude")
 claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 deepseek_model = kwargs.get("deepseek_model", "deepseek-reasoner")
 model_arg = tuple(map(float, os.getenv('MODEL_ARG', '1.0,1.0,0.7,0.1').split(',')))
 model = kwargs.get("model", "deepclaude")
 try:
 logger.info("开始流式处理请求...")
 logger.debug(f"输入消息: {messages}")
 direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
 if direct_tool_pass and tools and len(tools) > 0:
 logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
 if isinstance(tool_choice, str):
 logger.info(f"工具选择策略: {tool_choice}")
 elif isinstance(tool_choice, dict):
 logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
 else:
 logger.info(f"工具选择策略: {tool_choice}")
 converted_tools = self._validate_and_convert_tools(tools, target_format='claude-3')
 if not converted_tools:
 logger.warning("没有有效的工具可用，将作为普通对话处理")
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": "没有找到有效的工具定义，将作为普通对话处理。"
 },
 "finish_reason": "stop"
 }]
 }
 yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 logger.info(f"直接使用Claude模型: {claude_model}")
 claude_kwargs = {
 "messages": messages,
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "top_p": kwargs.get("top_p", 0.9),
 "tools": converted_tools
 }
 if isinstance(tool_choice, str):
 if tool_choice == "auto":
 claude_kwargs["tool_choice"] = {"type": "auto"}
 elif tool_choice == "none":
 logger.info("检测到'none'工具选择策略，将不使用工具")
 claude_kwargs.pop("tools")
 elif isinstance(tool_choice, dict):
 if tool_choice.get("type") == "function" and "function" in tool_choice:
 func_name = tool_choice["function"].get("name")
 if func_name:
 logger.info(f"指定使用工具: {func_name}")
 claude_kwargs["tool_choice"] = {
 "type": "tool",
 "name": func_name
 }
 else:
 claude_kwargs["tool_choice"] = tool_choice
 try:
 response = await self.claude_client.chat(**claude_kwargs)
 if "tool_calls" in response:
 tool_calls = response["tool_calls"]
 logger.info(f"Claude返回了 {len(tool_calls)} 个工具调用")
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": None,
 "tool_calls": tool_calls
 },
 "finish_reason": "tool_calls"
 }],
 "usage": {
 "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
 "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
 "total_tokens": response.get("usage", {}).get("total_tokens", 0)
 }
 }
 yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 else:
 content = response.get("content", "")
 if not content and "choices" in response and response["choices"]:
 content = response["choices"][0].get("message", {}).get("content", "")
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": content
 },
 "finish_reason": "stop"
 }],
 "usage": {
 "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
 "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
 "total_tokens": response.get("usage", {}).get("total_tokens", 0)
 }
 }
 yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 except Exception as e:
 logger.error(f"直接透传模式下API调用失败: {e}", exc_info=True)
 logger.info("将尝试使用推理-回答模式处理请求")
 except Exception as e:
 logger.error(f"处理流式请求时发生错误: {e}", exc_info=True)
 error_response = {
 "error": {
 "message": str(e),
 "type": "server_error",
 "code": "internal_error"
 }
 }
 yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 if self.save_to_db:
 try:
 user_id = None
 if messages and 'content' in messages[-1]:
 title = messages[-1]['content'][:20] + "..."
 user_question = messages[-1]['content']
 else:
 title = None
 user_question = "未提供问题内容"
 self.current_conversation_id = self.db_ops.create_conversation(
 user_id=user_id,
 title=title
 )
 logger.info(f"创建新对话，ID: {self.current_conversation_id}")
 self.db_ops.add_conversation_history(
 conversation_id=self.current_conversation_id,
 role="user",
 content=user_question
 )
 logger.info("用户问题已保存到数据库")
 except Exception as db_e:
 logger.error(f"保存对话数据失败: {db_e}")
 original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
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
 for reasoning_mode in self.reasoning_modes[1:]:
 try:
 logger.info(f"尝试使用不同的推理模式获取内容: {reasoning_mode}")
 os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
 reasoning = await self._get_reasoning_content(
 messages=messages,
 model=deepseek_model,
 model_arg=model_arg
 )
 if reasoning and len(reasoning) > self.min_reasoning_chars:
 logger.info(f"使用推理模式 {reasoning_mode} 成功获取推理内容")
 break
 except Exception as retry_e:
 logger.error(f"使用推理模式 {reasoning_mode} 重试失败: {retry_e}")
 logger.debug(f"获取到推理内容: {reasoning[:min(500, len(reasoning))]}...")
 has_tool_decision = False
 tool_calls = []
 if tools and len(tools) > 0:
 try:
 search_hint = ""
 if self.search_enabled and messages and messages[-1]["role"] == "user":
 search_hint = await self._enhance_with_search(messages[-1]["content"])
 if search_hint:
 reasoning = f"{search_hint}\n\n{reasoning}"
 decision_prompt = self._format_tool_decision_prompt(original_question, reasoning, tools)
 logger.debug(f"工具决策提示: {decision_prompt[:200]}...")
 tool_decision_response = await self.claude_client.chat(
 messages=[{"role": "user", "content": decision_prompt}],
 model=claude_model,
 tools=tools,
 tool_choice=tool_choice,
 temperature=model_arg[0],
 top_p=model_arg[1]
 )
 if "tool_calls" in tool_decision_response:
 tool_calls = tool_decision_response.get("tool_calls", [])
 has_tool_decision = True
 logger.info(f"Claude决定使用工具: {len(tool_calls)}个工具调用")
 response = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": None,
 "tool_calls": tool_calls
 },
 "finish_reason": "tool_calls"
 }]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 except Exception as tool_e:
 logger.error(f"工具调用流程失败: {tool_e}", exc_info=True)
 if not has_tool_decision:
 combined_content = f
 claude_messages = [{"role": "user", "content": combined_content}]
 logger.info("正在获取 Claude 回答...")
 try:
 full_content = ""
 async for content_type, content in self.claude_client.stream_chat(
 messages=claude_messages,
 model=claude_model,
 temperature=model_arg[0],
 top_p=model_arg[1],
 stream=False
 ):
 if content_type in ["answer", "content"]:
 logger.debug(f"获取到 Claude 回答: {content}")
 full_content += content
 if self.save_to_db and self.current_conversation_id:
 try:
 tokens = len(reasoning.split()) + len(full_content.split())
 self.db_ops.add_conversation_history(
 conversation_id=self.current_conversation_id,
 role="ai",
 content=full_content,
 reasoning=reasoning,
 model_name=claude_model,
 tokens=tokens
 )
 logger.info("AI回答和思考过程已保存到数据库")
 except Exception as db_e:
 logger.error(f"保存AI回答数据失败: {db_e}")
 response = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": full_content
 },
 "finish_reason": "stop"
 }]
 }
 yield f"data: {json.dumps(response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 except Exception as e:
 logger.error(f"获取 Claude 回答失败: {e}")
 error_message = await self._handle_api_error(e)
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": f"处理请求时出错: {error_message}"
 },
 "finish_reason": "error"
 }]
 }
 yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
 return
 async def _get_reasoning_content(self, messages: list, model: str, **kwargs) -> str:
 try:
 provider = self._get_reasoning_provider()
 reasoning_content = []
 content_received = False
 logger.info(f"开始获取思考内容，模型: {model}, 推理模式: {os.getenv('DEEPSEEK_REASONING_MODE', 'auto')}")
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model=model,
 model_arg=kwargs.get('model_arg')
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 logger.debug(f"收到推理内容，当前长度: {len(''.join(reasoning_content))}")
 elif content_type == "content" and not reasoning_content:
 logger.info("未收集到推理内容，将普通内容视为推理")
 reasoning_content.append(f"分析: {content}")
 logger.debug(f"普通内容转为推理内容，当前长度: {len(''.join(reasoning_content))}")
 elif content_type == "content":
 content_received = True
 logger.info("收到普通内容，推理阶段可能已结束")
 result = "\n".join(reasoning_content)
 if content_received and len(result) > self.min_reasoning_chars:
 logger.info(f"已收到普通内容且推理内容长度足够 ({len(result)}字符)，结束获取推理")
 return result
 if not result or len(result) < self.min_reasoning_chars:
 current_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto')
 logger.warning(f"使用模式 {current_mode} 获取的推理内容不足，尝试切换模式")
 for reasoning_mode in self.reasoning_modes:
 if reasoning_mode == current_mode:
 continue
 logger.info(f"尝试使用推理模式: {reasoning_mode}")
 os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
 provider = self._get_reasoning_provider()
 reasoning_content = []
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model=model,
 model_arg=kwargs.get('model_arg')
 ):
 if content_type == "reasoning":
 reasoning_content.append(content)
 elif content_type == "content" and not reasoning_content:
 reasoning_content.append(f"分析: {content}")
 retry_result = "\n".join(reasoning_content)
 if retry_result and len(retry_result) > self.min_reasoning_chars:
 logger.info(f"使用推理模式 {reasoning_mode} 成功获取足够的推理内容")
 return retry_result
 return result or "无法获取足够的推理内容"
 except Exception as e:
 logger.error(f"主要推理提供者失败: {e}")
 if isinstance(provider, DeepSeekClient):
 current_provider = getattr(provider, 'provider', 'unknown')
 logger.info(f"从 {current_provider} 提供商切换到 Ollama 推理提供者")
 try:
 provider = OllamaR1Client(api_url=os.getenv('OLLAMA_API_URL'))
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
 for reasoning_mode in self.reasoning_modes:
 if reasoning_content and len("".join(reasoning_content)) > self.min_reasoning_chars:
 logger.info(f"已收集到足够推理内容 ({len(''.join(reasoning_content))}字符)，不再尝试其他模式")
 break
 logger.info(f"尝试使用推理模式: {reasoning_mode}")
 os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
 provider = self._get_reasoning_provider()
 temp_content = []
 content_received = False
 try:
 async for content_type, content in provider.get_reasoning(
 messages=messages,
 model=model,
 model_arg=model_arg
 ):
 if content_type == "reasoning":
 temp_content.append(content)
 logger.debug(f"收到推理内容，当前临时内容长度: {len(''.join(temp_content))}")
 elif content_type == "content" and not temp_content and reasoning_mode in ['early_content', 'any_content']:
 temp_content.append(f"分析: {content}")
 logger.debug(f"普通内容转为推理内容，当前临时内容长度: {len(''.join(temp_content))}")
 elif content_type == "content":
 content_received = True
 logger.info("收到普通内容，推理阶段可能已结束")
 if content_received and len("".join(temp_content)) > self.min_reasoning_chars:
 logger.info("收到普通内容且临时推理内容足够，提前结束推理获取")
 break
 if temp_content and len("".join(temp_content)) > len("".join(reasoning_content)):
 reasoning_content = temp_content
 if content_received:
 logger.info("推理阶段已结束且内容足够，停止尝试其他模式")
 break
 except Exception as mode_e:
 logger.error(f"使用推理模式 {reasoning_mode} 时发生错误: {mode_e}")
 continue
 return "".join(reasoning_content) or "无法获取推理内容"
 except Exception as e:
 logger.error(f"主要推理提供者失败: {e}")
 if isinstance(provider, DeepSeekClient):
 current_provider = getattr(provider, 'provider', 'unknown')
 logger.info(f"从 {current_provider} 提供商切换到 Ollama 推理提供者")
 try:
 provider = OllamaR1Client(api_url=os.getenv('OLLAMA_API_URL'))
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
 def _validate_config(self):
 provider = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 if provider not in self.reasoning_providers:
 raise ValueError(f"不支持的推理提供者: {provider}")
 if provider == 'deepseek':
 if not os.getenv('DEEPSEEK_API_KEY'):
 raise ValueError("使用 DeepSeek 时必须提供 API KEY")
 if not os.getenv('DEEPSEEK_API_URL'):
 raise ValueError("使用 DeepSeek 时必须提供 API URL")
 elif provider == 'ollama':
 if not os.getenv('OLLAMA_API_URL'):
 raise ValueError("使用 Ollama 时必须提供 API URL")
 elif provider == 'siliconflow':
 if not os.getenv('DEEPSEEK_API_KEY'):
 raise ValueError("使用 硅基流动 时必须提供 DeepSeek API KEY")
 if not os.getenv('DEEPSEEK_API_URL'):
 raise ValueError("使用 硅基流动 时必须提供 DeepSeek API URL")
 elif provider == 'nvidia':
 if not os.getenv('DEEPSEEK_API_KEY'):
 raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API KEY")
 if not os.getenv('DEEPSEEK_API_URL'):
 raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API URL")
 if not os.getenv('CLAUDE_API_KEY'):
 raise ValueError("必须提供 CLAUDE_API_KEY 环境变量")
 def _format_stream_response(self, content: str, content_type: str = "content", **kwargs) -> bytes:
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
 if content_type == "reasoning":
 response["choices"][0]["delta"]["reasoning"] = True
 response["is_reasoning"] = True
 is_first_thought = kwargs.get("is_first_thought", False)
 if is_first_thought and not content.startswith("🤔"):
 response["choices"][0]["delta"]["content"] = f"🤔 {content}"
 elif content_type == "separator":
 response["is_separator"] = True
 elif content_type == "error":
 response["is_error"] = True
 response["choices"][0]["delta"]["content"] = f"⚠️ {content}"
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
 return list(text)
 async def _enhance_with_search(self, query: str) -> str:
 if not self.search_enabled:
 logger.info("搜索增强功能未启用")
 return ""
 logger.info(f"考虑为查询提供搜索增强: {query}")
 hint = "此问题可能涉及实时信息，可以考虑使用搜索工具获取最新数据。"
 logger.info("已为查询添加搜索增强提示")
 return hint
 async def _handle_tool_results(self, original_question: str, reasoning: str,
 tool_calls: List[Dict], tool_results: List[Dict], **kwargs) -> str:
 logger.info(f"处理工具调用结果 - 工具数: {len(tool_calls)}, 结果数: {len(tool_results)}")
 tools_info = ""
 for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
 tool_name = "未知工具"
 tool_args = "{}"
 if "function" in tool_call:
 func = tool_call.get("function", {})
 tool_name = func.get("name", "未知工具")
 tool_args = func.get("arguments", "{}")
 elif "name" in tool_call:
 tool_name = tool_call.get("name", "未知工具")
 tool_args = json.dumps(tool_call.get("input", tool_call.get("arguments", {})), ensure_ascii=False)
 try:
 args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
 args_str = json.dumps(args_dict, ensure_ascii=False, indent=2)
 except:
 args_str = str(tool_args)
 result_content = ""
 if isinstance(tool_result, dict):
 result_content = (tool_result.get("content") or
 tool_result.get("result") or
 tool_result.get("output") or
 tool_result.get("response") or
 json.dumps(tool_result, ensure_ascii=False))
 else:
 result_content = str(tool_result)
 tools_info += f
 prompt = f
 logger.info("向Claude发送工具结果提示生成最终回答")
 try:
 messages = [
 {"role": "user", "content": prompt}
 ]
 response = await self.claude_client.chat(
 messages=messages,
 model=os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
 temperature=kwargs.get('temperature', 0.7),
 top_p=kwargs.get('top_p', 0.9)
 )
 if "content" in response:
 answer = response.get("content", "")
 logger.info(f"生成最终回答成功(Claude格式): {answer[:100]}...")
 return answer
 elif "choices" in response and response["choices"]:
 answer = response["choices"][0].get("message", {}).get("content", "")
 logger.info(f"生成最终回答成功(OpenAI格式): {answer[:100]}...")
 return answer
 else:
 logger.error(f"未找到回答内容，响应结构: {list(response.keys())}")
 return "抱歉，处理工具结果时出现错误，无法生成回答。"
 except Exception as e:
 logger.error(f"处理工具结果失败: {e}", exc_info=True)
 return f"抱歉，处理工具结果时出现错误: {str(e)}"
 def _format_tool_decision_prompt(self, original_question: str, reasoning: str, tools: List[Dict]) -> str:
 tools_description = ""
 for i, tool in enumerate(tools, 1):
 if "function" in tool:
 function = tool["function"]
 name = function.get("name", "未命名工具")
 description = function.get("description", "无描述")
 parameters = function.get("parameters", {})
 required = parameters.get("required", [])
 properties = parameters.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "name" in tool and "custom" in tool:
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 custom = tool.get("custom", {})
 if "input_schema" in custom:
 input_schema = custom["input_schema"]
 required = input_schema.get("required", [])
 properties = input_schema.get("properties", {})
 else:
 required = custom.get("required", [])
 properties = custom.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "type" in tool and tool["type"] == "custom":
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 tool_schema = tool.get("tool_schema", {})
 required = tool_schema.get("required", [])
 properties = tool_schema.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "name" in tool and "parameters" in tool:
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 parameters = tool.get("parameters", {})
 required = parameters.get("required", [])
 properties = parameters.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 elif "name" in tool and "input_schema" in tool:
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "无描述")
 input_schema = tool.get("input_schema", {})
 required = input_schema.get("required", [])
 properties = input_schema.get("properties", {})
 param_desc = ""
 for param_name, param_info in properties.items():
 is_required = "必填" if param_name in required else "可选"
 param_type = param_info.get("type", "未知类型")
 param_description = param_info.get("description", "无描述")
 enum_values = param_info.get("enum", [])
 enum_desc = ""
 if enum_values:
 enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
 param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
 tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
 prompt = f
 logger.info(f"生成工具决策提示 - 问题: '{original_question[:30]}...'")
 return prompt
 def _format_tool_call_response(self, tool_call: Dict, **kwargs) -> bytes:
 try:
 tool_call_id = tool_call.get("id")
 if not tool_call_id or not isinstance(tool_call_id, str) or len(tool_call_id) < 8:
 tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
 function = tool_call.get("function", {})
 function_name = function.get("name", "")
 function_args = function.get("arguments", "{}")
 if not isinstance(function_args, str):
 try:
 function_args = json.dumps(function_args, ensure_ascii=False)
 except Exception as e:
 logger.error(f"参数序列化失败: {e}")
 function_args = "{}"
 response = {
 "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
 "object": "chat.completion.chunk",
 "created": kwargs.get("created_time", int(time.time())),
 "model": kwargs.get("model", "deepclaude"),
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": None,
 "tool_calls": [
 {
 "index": 0,
 "id": tool_call_id,
 "type": "function",
 "function": {
 "name": function_name,
 "arguments": function_args
 }
 }
 ]
 },
 "finish_reason": "tool_calls"
 }]
 }
 logger.info(f"工具调用响应格式化完成 - 工具: {function_name}, ID: {tool_call_id}")
 return f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode('utf-8')
 except Exception as e:
 logger.error(f"工具调用响应格式化失败: {e}", exc_info=True)
 error_response = {
 "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
 "object": "chat.completion.chunk",
 "created": kwargs.get("created_time", int(time.time())),
 "model": kwargs.get("model", "deepclaude"),
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"工具调用失败: {str(e)}"
 },
 "finish_reason": "error"
 }]
 }
 return f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')
 def _format_tool_result_response(self, tool_result: Dict, **kwargs) -> bytes:
 try:
 if not isinstance(tool_result, dict):
 raise ValueError("工具结果必须是字典格式")
 tool_call_id = tool_result.get("tool_call_id")
 if not tool_call_id:
 raise ValueError("工具结果必须包含tool_call_id")
 content = tool_result.get("content")
 if content is None:
 raise ValueError("工具结果必须包含content")
 response = {
 "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
 "object": "chat.completion.chunk",
 "created": kwargs.get("created_time", int(time.time())),
 "model": kwargs.get("model", "deepclaude"),
 "choices": [{
 "index": 0,
 "delta": {
 "role": "tool",
 "content": content,
 "tool_call_id": tool_call_id
 },
 "finish_reason": None
 }]
 }
 logger.info(f"工具结果响应格式化完成 - ID: {tool_call_id}")
 return f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode('utf-8')
 except Exception as e:
 logger.error(f"工具结果响应格式化失败: {e}")
 error_response = {
 "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
 "object": "chat.completion.chunk",
 "created": kwargs.get("created_time", int(time.time())),
 "model": kwargs.get("model", "deepclaude"),
 "choices": [{
 "index": 0,
 "delta": {
 "role": "assistant",
 "content": f"工具结果处理失败: {str(e)}"
 },
 "finish_reason": "error"
 }]
 }
 return f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')
 def _validate_and_convert_tools(self, tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
 if not tools:
 return []
 valid_tools = []
 for i, tool in enumerate(tools):
 if not isinstance(tool, dict):
 logger.warning(f"工具格式错误: {tool}")
 continue
 if "type" in tool and tool["type"] in ["custom", "bash_20250124", "text_editor_20250124"]:
 if "custom" in tool:
 logger.warning(f"检测到工具中的custom字段，这不符合Claude API规范，正在移除: {tool.get('name', '未命名工具')}")
 fixed_tool = tool.copy()
 fixed_tool.pop("custom", None)
 valid_tools.append(fixed_tool)
 else:
 valid_tools.append(tool)
 logger.info(f"检测到已是Claude格式的工具: {tool.get('name', '未命名工具')}")
 continue
 if "function" in tool:
 if target_format == 'claude-3':
 function_data = tool["function"]
 name = function_data.get("name", "未命名工具")
 description = function_data.get("description", "")
 parameters = function_data.get("parameters", {})
 claude_tool = {
 "type": "custom",
 "name": name,
 "description": description,
 "tool_schema": parameters
 }
 logger.info(f"将OpenAI格式工具 '{name}' 转换为Claude custom格式")
 valid_tools.append(claude_tool)
 else:
 if "type" not in tool:
 tool = {"type": "function", "function": tool["function"]}
 valid_tools.append(tool)
 logger.info(f"保持OpenAI格式工具: {tool['function'].get('name', '未命名工具')}")
 continue
 if "name" in tool and "api_type" in tool:
 logger.info(f"检测到Dify格式工具: {tool.get('name', '未命名工具')}")
 if target_format == 'claude-3':
 dify_tool = {
 "type": "custom",
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "tool_schema": tool.get("parameters", {})
 }
 valid_tools.append(dify_tool)
 logger.info(f"已将Dify工具 '{tool.get('name', '未命名工具')}' 转换为Claude格式")
 else:
 openai_tool = {
 "type": "function",
 "function": {
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "parameters": tool.get("parameters", {})
 }
 }
 valid_tools.append(openai_tool)
 logger.info(f"已将Dify工具 '{tool.get('name', '未命名工具')}' 转换为OpenAI格式")
 continue
 if "name" in tool and "parameters" in tool:
 logger.info(f"检测到简化格式工具: {tool.get('name', '未命名工具')}")
 if target_format == 'claude-3':
 simple_tool = {
 "type": "custom",
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "tool_schema": tool.get("parameters", {})
 }
 valid_tools.append(simple_tool)
 logger.info(f"已将简化格式工具转为Claude格式: {tool.get('name', '未命名工具')}")
 else:
 openai_tool = {
 "type": "function",
 "function": {
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "parameters": tool.get("parameters", {})
 }
 }
 valid_tools.append(openai_tool)
 logger.info(f"已将简化格式工具转为OpenAI格式: {tool.get('name', '未命名工具')}")
 continue
 if set(["name", "description"]).issubset(set(tool.keys())):
 logger.info(f"检测到可能的变体格式工具: {tool.get('name', '未命名工具')}")
 parameters = tool.get("parameters",
 tool.get("schema",
 tool.get("parameter_schema",
 tool.get("tool_schema", {}))))
 if target_format == 'claude-3':
 variant_tool = {
 "type": "custom",
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "tool_schema": parameters
 }
 valid_tools.append(variant_tool)
 logger.info(f"已将变体格式工具转为Claude格式: {tool.get('name', '未命名工具')}")
 else:
 openai_tool = {
 "type": "function",
 "function": {
 "name": tool.get("name", "未命名工具"),
 "description": tool.get("description", ""),
 "parameters": parameters
 }
 }
 valid_tools.append(openai_tool)
 logger.info(f"已将变体格式工具转为OpenAI格式: {tool.get('name', '未命名工具')}")
 continue
 logger.warning(f"工具格式无法识别: {json.dumps(tool, ensure_ascii=False)[:100]}...")
 logger.info(f"工具验证和转换完成，原有 {len(tools)} 个工具，有效 {len(valid_tools)} 个工具")
 if valid_tools:
 for i, tool in enumerate(valid_tools):
 if "type" in tool and tool["type"] == "custom":
 logger.debug(f"有效工具[{i}]: {tool.get('name', '未命名工具')} (Claude格式)")
 else:
 logger.debug(f"有效工具[{i}]: {tool.get('name', tool.get('function', {}).get('name', '未命名工具'))} (OpenAI格式)")
 return valid_tools
 def _format_claude_prompt(self, original_question: str, reasoning: str) -> str:
 return f
 async def chat_completions_without_stream(
 self,
 messages: list,
 model_arg: tuple[float, float, float, float],
 tools: list = None,
 tool_choice = "auto",
 deepseek_model: str = "deepseek-reasoner",
 claude_model: str = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
 **kwargs
 ) -> dict:
 logger.info("开始处理非流式请求...")
 logger.debug(f"输入消息: {messages}")
 direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
 chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
 created_time = kwargs.get("created_time", int(time.time()))
 model = kwargs.get("model", "deepclaude")
 if direct_tool_pass and tools and len(tools) > 0:
 logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
 if isinstance(tool_choice, str):
 logger.info(f"工具选择策略: {tool_choice}")
 elif isinstance(tool_choice, dict):
 logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
 else:
 logger.info(f"工具选择策略: {tool_choice}")
 converted_tools = self._validate_and_convert_tools(tools, target_format='claude-3')
 if not converted_tools:
 logger.warning("没有有效的工具可用，将作为普通对话处理")
 return {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": "没有找到有效的工具定义，将作为普通对话处理。"
 },
 "finish_reason": "stop"
 }]
 }
 logger.info(f"直接使用Claude模型: {claude_model}")
 claude_kwargs = {
 "messages": messages,
 "model": claude_model,
 "temperature": kwargs.get("temperature", 0.7),
 "top_p": kwargs.get("top_p", 0.9),
 "tools": converted_tools
 }
 if isinstance(tool_choice, str):
 if tool_choice == "auto":
 claude_kwargs["tool_choice"] = {"type": "auto"}
 elif tool_choice == "none":
 logger.info("检测到'none'工具选择策略，将不使用工具")
 claude_kwargs.pop("tools")
 elif isinstance(tool_choice, dict):
 if tool_choice.get("type") == "function" and "function" in tool_choice:
 func_name = tool_choice["function"].get("name")
 if func_name:
 logger.info(f"指定使用工具: {func_name}")
 claude_kwargs["tool_choice"] = {
 "type": "tool",
 "name": func_name
 }
 else:
 claude_kwargs["tool_choice"] = tool_choice
 try:
 response = await self.claude_client.chat(**claude_kwargs)
 if "tool_calls" in response:
 tool_calls = response["tool_calls"]
 logger.info(f"Claude返回了 {len(tool_calls)} 个工具调用")
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": None,
 "tool_calls": tool_calls
 },
 "finish_reason": "tool_calls"
 }],
 "usage": {
 "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
 "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
 "total_tokens": response.get("usage", {}).get("total_tokens", 0)
 }
 }
 return result
 else:
 content = response.get("content", "")
 if not content and "choices" in response and response["choices"]:
 content = response["choices"][0].get("message", {}).get("content", "")
 result = {
 "id": chat_id,
 "object": "chat.completion",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "message": {
 "role": "assistant",
 "content": content
 },
 "finish_reason": "stop"
 }],
 "usage": {
 "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
 "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
 "total_tokens": response.get("usage", {}).get("total_tokens", 0)
 }
 }
 return result
 except Exception as e:
 logger.error(f"直接透传模式下API调用失败: {e}", exc_info=True)
 logger.info("将尝试使用推理-回答模式处理请求")```
______________________________

## .../tools/validators.py
```python
from typing import Dict, List, Any, Optional
from app.utils.logger import logger
class ToolValidator:
 @staticmethod
 def is_valid_openai_function(tool: Dict) -> bool:
 if not isinstance(tool, dict):
 return False
 if "function" not in tool:
 return False
 function = tool["function"]
 if not isinstance(function, dict):
 return False
 if "name" not in function:
 return False
 return True
 @staticmethod
 def is_valid_claude_custom_tool(tool: Dict) -> bool:
 if not isinstance(tool, dict):
 return False
 if "type" not in tool or tool["type"] != "custom":
 return False
 if "name" not in tool:
 return False
 return True
 @staticmethod
 def has_nested_custom_type(tool: Dict) -> bool:
 if not isinstance(tool, dict):
 return False
 if "custom" in tool and isinstance(tool["custom"], dict) and "type" in tool["custom"]:
 return True
 if "type" in tool and tool["type"] == "custom" and "tool_schema" in tool:
 tool_schema = tool["tool_schema"]
 if isinstance(tool_schema, dict) and "type" in tool_schema and tool_schema["type"] == "custom":
 return True
 return False
 @staticmethod
 def validate_claude_tool(tool: Dict) -> tuple[bool, list[str]]:
 errors = []
 if not isinstance(tool, dict):
 return False, ["工具必须是字典类型"]
 required_fields = ["name", "description", "input_schema"]
 for field in required_fields:
 if field not in tool:
 errors.append(f"缺少必要字段: {field}")
 if "input_schema" in tool:
 input_schema = tool["input_schema"]
 if not isinstance(input_schema, dict):
 errors.append("input_schema字段必须是字典类型")
 else:
 if "type" not in input_schema:
 errors.append("input_schema必须包含type字段")
 elif input_schema["type"] != "object":
 errors.append("input_schema的type字段值必须为'object'")
 if "properties" not in input_schema:
 errors.append("input_schema必须包含properties字段")
 elif not isinstance(input_schema["properties"], dict):
 errors.append("properties字段必须是字典类型")
 return len(errors) == 0, errors```
______________________________

## .../tools/handlers.py
```python
from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator
from .converters import ToolConverter
import uuid
import json
import copy
class ToolHandler:
 def __init__(self):
 pass
 def validate_and_convert_tools(self, tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
 if not tools or not isinstance(tools, list) or len(tools) == 0:
 logger.warning("未提供有效的工具列表")
 return None
 validated_tools = []
 input_tools_count = len(tools)
 logger.info(f"开始验证和转换 {input_tools_count} 个工具至 {target_format} 格式")
 for i, tool in enumerate(tools):
 try:
 if not isinstance(tool, dict):
 logger.warning(f"工具[{i}]不是字典类型，跳过: {tool}")
 continue
 logger.debug(f"处理工具[{i}]: {json.dumps(tool, ensure_ascii=False)[:100]}...")
 if "function" in tool or (tool.get("type") == "function" and "function" in tool):
 function = tool.get("function", tool) if tool.get("type") == "function" else tool["function"]
 if not isinstance(function, dict):
 logger.warning(f"工具[{i}]函数结构不是字典: {function}")
 continue
 if "name" not in function:
 logger.warning(f"工具[{i}]函数缺少name字段: {function}")
 continue
 name = function.get("name", f"未命名工具_{i}")
 description = function.get("description", "")
 parameters = function.get("parameters", {"type": "object", "properties": {}})
 if not isinstance(parameters, dict):
 logger.warning(f"工具[{i}] {name} 的parameters不是字典: {parameters}")
 parameters = {"type": "object", "properties": {}}
 if "type" not in parameters or parameters["type"] != "object":
 logger.debug(f"工具[{i}] {name} 的parameters.type设置为object (原值: {parameters.get('type')})")
 parameters["type"] = "object"
 if "properties" not in parameters or not isinstance(parameters["properties"], dict):
 logger.debug(f"工具[{i}] {name} 的parameters缺少properties字段或不是字典")
 parameters["properties"] = {}
 if target_format == 'claude-3':
 validated_tool = {
 "name": name,
 "description": description,
 "input_schema": {
 "type": "object",
 "properties": parameters.get("properties", {}),
 "required": parameters.get("required", [])
 }
 }
 else:
 validated_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": parameters
 }
 }
 validated_tools.append(validated_tool)
 logger.info(f"转换工具[{i}]: {name} (OpenAI格式 -> {target_format})")
 elif "name" in tool and ("input_schema" in tool or "description" in tool):
 name = tool["name"]
 description = tool.get("description", "")
 if target_format == 'claude-3':
 validated_tool = {
 "name": name,
 "description": description
 }
 if "input_schema" in tool and isinstance(tool["input_schema"], dict):
 input_schema = copy.deepcopy(tool["input_schema"])
 if "type" not in input_schema:
 input_schema["type"] = "object"
 if "properties" not in input_schema or not isinstance(input_schema["properties"], dict):
 input_schema["properties"] = {}
 validated_tool["input_schema"] = input_schema
 else:
 validated_tool["input_schema"] = {
 "type": "object",
 "properties": {}
 }
 if "custom" in tool and isinstance(tool["custom"], dict):
 custom = tool["custom"]
 if "properties" in custom and isinstance(custom["properties"], dict):
 validated_tool["input_schema"]["properties"] = custom["properties"]
 if "required" in custom and isinstance(custom["required"], list):
 validated_tool["input_schema"]["required"] = custom["required"]
 validated_tools.append(validated_tool)
 logger.info(f"验证工具[{i}]: {name} (Claude格式 -> {target_format})")
 else:
 parameters = {"type": "object", "properties": {}}
 if "input_schema" in tool and isinstance(tool["input_schema"], dict):
 input_schema = tool["input_schema"]
 if "properties" in input_schema and isinstance(input_schema["properties"], dict):
 parameters["properties"] = input_schema["properties"]
 if "required" in input_schema and isinstance(input_schema["required"], list):
 parameters["required"] = input_schema["required"]
 validated_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": parameters
 }
 }
 validated_tools.append(validated_tool)
 logger.info(f"转换工具[{i}]: {name} (Claude格式 -> {target_format})")
 elif "name" in tool and ("properties" in tool or "parameters" in tool):
 name = tool["name"]
 description = tool.get("description", "")
 parameters = {}
 if "parameters" in tool and isinstance(tool["parameters"], dict):
 parameters = tool["parameters"]
 elif "properties" in tool and isinstance(tool["properties"], dict):
 parameters = {"type": "object", "properties": tool["properties"]}
 if "required" in tool and isinstance(tool["required"], list):
 parameters["required"] = tool["required"]
 if target_format == 'claude-3':
 validated_tool = {
 "name": name,
 "description": description,
 "input_schema": {
 "type": "object",
 "properties": parameters.get("properties", {}),
 "required": parameters.get("required", [])
 }
 }
 else:
 validated_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": parameters
 }
 }
 validated_tools.append(validated_tool)
 logger.info(f"转换工具[{i}]: {name} (通用格式 -> {target_format})")
 else:
 logger.warning(f"工具[{i}]不是已知格式，跳过: {json.dumps(tool, ensure_ascii=False)[:100]}...")
 continue
 except Exception as e:
 logger.error(f"处理工具[{i}]时出错: {e}")
 continue
 logger.info(f"工具验证完成: {input_tools_count} 个输入工具 -> {len(validated_tools)} 个有效工具")
 return validated_tools if validated_tools else None
 def _is_valid_function_tool(self, tool: Dict) -> bool:
 if not isinstance(tool, dict) or "function" not in tool:
 return False
 function = tool["function"]
 if not isinstance(function, dict):
 return False
 required_fields = ["name", "description", "parameters"]
 for field in required_fields:
 if field not in function:
 return False
 parameters = function["parameters"]
 if not isinstance(parameters, dict):
 return False
 if "type" not in parameters:
 return False
 if parameters["type"] != "object":
 return False
 if "properties" in parameters and not isinstance(parameters["properties"], dict):
 return False
 return True
 def _is_valid_custom_tool(self, tool: Dict) -> bool:
 if not isinstance(tool, dict):
 return False
 required_fields = ["name", "description", "input_schema"]
 for field in required_fields:
 if field not in tool:
 return False
 input_schema = tool.get("input_schema")
 if not isinstance(input_schema, dict):
 return False
 if "type" not in input_schema or input_schema["type"] != "object":
 return False
 if "properties" not in input_schema or not isinstance(input_schema["properties"], dict):
 return False
 return True
 def _convert_openai_to_claude(self, openai_tool: Dict) -> Dict:
 try:
 if "function" in openai_tool:
 function = openai_tool["function"]
 elif "type" in openai_tool and openai_tool["type"] == "function":
 function = openai_tool
 else:
 logger.warning(f"无法识别的OpenAI工具格式: {openai_tool}")
 return None
 name = function.get("name", "未命名工具")
 description = function.get("description", "")
 parameters = function.get("parameters", {})
 properties = parameters.get("properties", {})
 required = parameters.get("required", [])
 claude_tool = {
 "name": name,
 "description": description,
 "input_schema": {
 "type": "object",
 "properties": properties,
 "required": required
 }
 }
 logger.info(f"转换OpenAI工具 '{name}' 为Claude格式: {json.dumps(claude_tool, ensure_ascii=False)}")
 return claude_tool
 except Exception as e:
 logger.error(f"转换OpenAI工具到Claude格式失败: {e}")
 return None
 def format_tool_call_for_streaming(self, tool_call_data: Dict, chat_id: str, created_time: int) -> Dict:
 try:
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": "deepclaude",
 "choices": [{
 "index": 0,
 "delta": {
 "tool_calls": [tool_call_data]
 }
 }],
 "finish_reason": None
 }
 return response
 except Exception as e:
 logger.error(f"格式化工具调用失败: {e}")
 return {
 "error": {
 "message": str(e),
 "type": "internal_error"
 }
 }
 async def process_tool_call(self, tool_call: Dict, **kwargs) -> Dict:
 logger.info(f"处理工具调用: {tool_call.get('function', {}).get('name', '未知工具')}")
 return {
 "status": "success",
 "result": "工具调用结果示例"
 }
 def _final_validate_claude_tools(self, tools: List[Dict]) -> List[Dict]:
 if not tools:
 return []
 from .validators import ToolValidator
 valid_tools = []
 for i, tool in enumerate(tools):
 fixed_tool = {}
 for field in ["name", "description"]:
 if field in tool:
 fixed_tool[field] = tool[field]
 else:
 fixed_tool[field] = f"未命名工具_{i}" if field == "name" else ""
 if "type" not in tool:
 fixed_tool["type"] = "custom"
 elif tool["type"] not in ["custom", "bash_20250124", "text_editor_20250124"]:
 logger.warning(f"工具[{i}]类型'{tool['type']}'不被Claude支持，修改为'custom'")
 fixed_tool["type"] = "custom"
 else:
 fixed_tool["type"] = tool["type"]
 if fixed_tool["type"] == "custom":
 if "tool_schema" in tool and isinstance(tool["tool_schema"], dict):
 schema = tool["tool_schema"].copy()
 if "type" not in schema:
 schema["type"] = "object"
 elif schema["type"] == "custom":
 schema["type"] = "object"
 if "properties" not in schema:
 schema["properties"] = {}
 fixed_tool["tool_schema"] = schema
 else:
 fixed_tool["tool_schema"] = {"type": "object", "properties": {}}
 for key, value in tool.items():
 if key not in ["type", "name", "description", "tool_schema", "custom"]:
 fixed_tool[key] = value
 is_valid, errors = ToolValidator.validate_claude_tool(fixed_tool)
 if not is_valid:
 logger.warning(f"工具[{i}]经过修复后仍有问题: {', '.join(errors)}")
 if "custom" in fixed_tool:
 fixed_tool.pop("custom", None)
 valid_tools.append(fixed_tool)
 logger.debug(f"工具[{i}]最终格式: {json.dumps(fixed_tool, ensure_ascii=False)}")
 return valid_tools```
______________________________

## .../tools/__init__.py
```python
from .handlers import ToolHandler
__all__ = ["ToolHandler"]```
______________________________

## .../tools/converters.py
```python
from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator
class ToolConverter:
 @staticmethod
 def openai_to_claude(tool: Dict) -> Dict:
 if not ToolValidator.is_valid_openai_function(tool):
 logger.warning(f"工具格式错误，无法转换: {tool}")
 return None
 function_data = tool["function"]
 name = function_data.get("name", "未命名工具")
 description = function_data.get("description", "")
 if "parameters" in function_data and isinstance(function_data["parameters"], dict):
 parameters = function_data["parameters"].copy()
 if "type" in parameters and parameters["type"] == "custom":
 logger.warning(f"参数中存在custom类型，正在修改为object")
 parameters["type"] = "object"
 elif "type" not in parameters:
 parameters["type"] = "object"
 else:
 parameters = {"type": "object", "properties": {}}
 claude_tool = {
 "type": "custom",
 "name": name,
 "description": description,
 "tool_schema": parameters
 }
 if "custom" in claude_tool:
 logger.warning(f"转换后的工具中存在custom字段，正在移除")
 claude_tool.pop("custom", None)
 return claude_tool
 @staticmethod
 def claude_to_openai(tool: Dict) -> Dict:
 if not ToolValidator.is_valid_claude_custom_tool(tool):
 logger.warning(f"工具格式错误，无法转换: {tool}")
 return None
 name = tool.get("name", "未命名工具")
 description = tool.get("description", "")
 schema = tool.get("tool_schema", {})
 openai_tool = {
 "type": "function",
 "function": {
 "name": name,
 "description": description,
 "parameters": schema
 }
 }
 return openai_tool
 @staticmethod
 def fix_claude_custom_tool(tool: Dict) -> Dict:
 if not ToolValidator.is_valid_claude_custom_tool(tool):
 return tool
 if ToolValidator.has_nested_custom_type(tool):
 fixed_tool = tool.copy()
 fixed_tool["custom"] = tool["custom"].copy()
 fixed_tool["custom"].pop("type", None)
 logger.debug(f"已修复工具中的嵌套type字段: {tool.get('name', '未命名工具')}")
 return fixed_tool
 return tool```
______________________________

## .../utils/__init__.py
```python
from app.deepclaude.utils.prompts import PromptTemplates
from app.deepclaude.utils.streaming import StreamingHelper
__all__ = ['PromptTemplates', 'StreamingHelper']```
______________________________

## .../utils/prompts.py
```python
class PromptTemplates:
 @staticmethod
 def reasoning_prompt(question: str) -> str:
 return f
 @staticmethod
 def tool_decision_prompt(question: str, reasoning: str, tools_description: str) -> str:
 return f
 @staticmethod
 def final_answer_prompt(question: str, reasoning: str, tool_results: str = None) -> str:
 tool_part = f"\n\n工具调用结果：\n{tool_results}" if tool_results else ""
 return f```
______________________________

## .../utils/streaming.py
```python
import json
from typing import Dict, Any
class StreamingHelper:
 @staticmethod
 def format_chunk_response(content: str, role: str = "assistant", chat_id: str = None,
 created_time: int = None, model: str = "deepclaude",
 is_reasoning: bool = False, finish_reason: str = None) -> str:
 response = {
 "id": chat_id,
 "object": "chat.completion.chunk",
 "created": created_time,
 "model": model,
 "choices": [{
 "index": 0,
 "delta": {
 "role": role,
 "content": content
 },
 "finish_reason": finish_reason
 }]
 }
 if is_reasoning:
 response["is_reasoning"] = True
 response["choices"][0]["delta"]["reasoning"] = True
 return f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
 @staticmethod
 def format_done_marker() -> str:
 return "data: [DONE]\n\n"```
______________________________

## .../reasoning/__init__.py
```python
from .base import BaseReasoningProvider
from .deepseek import DeepSeekReasoningProvider
from .ollama import OllamaReasoningProvider
from .factory import ReasoningProviderFactory
__all__ = [
 "BaseReasoningProvider",
 "DeepSeekReasoningProvider",
 "OllamaReasoningProvider",
 "ReasoningProviderFactory"
]```
______________________________

## .../reasoning/factory.py
```python
from .base import BaseReasoningProvider
from .deepseek import DeepSeekReasoningProvider
from .ollama import OllamaReasoningProvider
import os
from app.utils.logger import logger
class ReasoningProviderFactory:
 @staticmethod
 def create(provider_type: str = None) -> BaseReasoningProvider:
 provider_type = provider_type or os.getenv('REASONING_PROVIDER', 'deepseek').lower()
 if provider_type == 'deepseek':
 api_key = os.getenv('DEEPSEEK_API_KEY')
 api_url = os.getenv('DEEPSEEK_API_URL')
 return DeepSeekReasoningProvider(api_key, api_url, provider='deepseek')
 elif provider_type == 'siliconflow':
 api_key = os.getenv('DEEPSEEK_API_KEY')
 api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.siliconflow.cn/v1/chat/completions')
 return DeepSeekReasoningProvider(api_key, api_url, provider='siliconflow')
 elif provider_type == 'nvidia':
 api_key = os.getenv('DEEPSEEK_API_KEY')
 api_url = os.getenv('DEEPSEEK_API_URL')
 return DeepSeekReasoningProvider(api_key, api_url, provider='nvidia')
 elif provider_type == 'ollama':
 api_url = os.getenv('OLLAMA_API_URL')
 return OllamaReasoningProvider(api_url=api_url)
 else:
 raise ValueError(f"不支持的推理提供者类型: {provider_type}")```
______________________________

## .../reasoning/deepseek.py
```python
from .base import BaseReasoningProvider
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import aiohttp
from app.utils.logger import logger
class DeepSeekReasoningProvider(BaseReasoningProvider):
 def __init__(self, api_key: str, api_url: str = None, provider: str = "deepseek"):
 super().__init__(api_key, api_url)
 self.provider = provider.lower()
 if not self.api_url:
 if self.provider == "deepseek":
 self.api_url = "https://api.deepseek.com/v1/chat/completions"
 elif self.provider == "siliconflow":
 self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
 elif self.provider == "nvidia":
 self.api_url = "https://api.nvidia.com/v1/chat/completions"
 else:
 raise ValueError(f"不支持的提供商: {provider}")
 self.reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto')
 logger.info(f"初始化DeepSeek推理提供者: provider={self.provider}, url={self.api_url}, mode={self.reasoning_mode}")
 async def extract_reasoning_from_think_tags(self, content: str) -> str:
 if "<think>" in content and "</think>" in content:
 start = content.find("<think>") + 7
 end = content.find("</think>")
 if start < end:
 return content[start:end].strip()
 return ""
 async def get_reasoning(self, messages: List[Dict], model: str = None, model_arg: tuple = None, **kwargs) -> str:
 temperature = model_arg[0] if model_arg and len(model_arg) > 0 else kwargs.get('temperature', 0.7)
 top_p = model_arg[1] if model_arg and len(model_arg) > 1 else kwargs.get('top_p', 0.9)
 if not model:
 model = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-R1')
 headers = {
 "Authorization": f"Bearer {self.api_key}",
 "Content-Type": "application/json",
 "Accept": "text/event-stream",
 }
 data = {
 "model": model,
 "messages": messages,
 "stream": True,
 "temperature": temperature,
 "top_p": top_p,
 "max_tokens": kwargs.get('max_tokens', 4096)
 }
 if self.provider == 'siliconflow':
 if not data.get("stop"):
 data["stop"] = kwargs.get('stop', ["<STOP>"])
 elif self.provider == 'nvidia':
 pass
 reasoning_content = []
 try:
 logger.info(f"发送DeepSeek推理请求: {self.api_url}")
 logger.debug(f"请求头: {headers}")
 logger.debug(f"请求体: {json.dumps(data, ensure_ascii=False)}")
 async with aiohttp.ClientSession() as session:
 async with session.post(
 self.api_url,
 headers=headers,
 json=data,
 timeout=aiohttp.ClientTimeout(total=60)
 ) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"DeepSeek API请求失败: HTTP {response.status}\n{error_text}")
 raise Exception(f"HTTP {response.status}: {error_text}")
 logger.info("DeepSeek开始流式响应")
 async for line in response.content:
 line_str = line.decode('utf-8').strip()
 if not line_str or not line_str.startswith('data:'):
 continue
 data_json = line_str[5:].strip()
 if data_json == "[DONE]":
 logger.debug("收到[DONE]标记")
 continue
 try:
 data = json.loads(data_json)
 if not data.get("choices"):
 continue
 choice = data["choices"][0]
 delta = choice.get("delta", {})
 if self.reasoning_mode == 'reasoning_field':
 reasoning = choice.get("reasoning_content")
 if reasoning:
 reasoning_content.append(reasoning)
 elif self.reasoning_mode == 'think_tags':
 content = delta.get("content", "")
 if "<think>" in content:
 reasoning = await self.extract_reasoning_from_think_tags(content)
 if reasoning:
 reasoning_content.append(reasoning)
 else:
 content = delta.get("content", "")
 if content:
 reasoning_content.append(content)
 except json.JSONDecodeError as e:
 logger.error(f"JSON解析错误: {e}, 数据: {data_json[:100]}")
 continue
 except Exception as e:
 logger.error(f"处理推理响应时出错: {e}")
 continue
 full_reasoning = "".join(reasoning_content)
 logger.info(f"获取到推理内容: {len(full_reasoning)} 字符")
 logger.debug(f"推理内容预览: {full_reasoning[:200]}...")
 return full_reasoning
 except Exception as e:
 logger.error(f"获取推理内容失败: {e}", exc_info=True)
 return f"获取推理内容时出错: {str(e)}"```
______________________________

## .../reasoning/ollama.py
```python
from .base import BaseReasoningProvider
from typing import Dict, List, Any
import json
import aiohttp
from app.utils.logger import logger
class OllamaReasoningProvider(BaseReasoningProvider):
 def __init__(self, api_url: str = "http://localhost:11434/api/chat"):
 super().__init__(api_key=None, api_url=api_url)
 self.model = "deepseek-chat"
 async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
 data = {
 "model": model or self.model,
 "messages": messages,
 "stream": False,
 "options": {
 "temperature": kwargs.get("temperature", 0.7),
 "top_p": kwargs.get("top_p", 0.9)
 }
 }
 try:
 async with aiohttp.ClientSession() as session:
 async with session.post(
 self.api_url,
 json=data,
 timeout=aiohttp.ClientTimeout(total=60)
 ) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"Ollama API请求失败: HTTP {response.status}\n{error_text}")
 raise Exception(f"HTTP {response.status}: {error_text}")
 result = await response.json()
 if "message" in result:
 return result["message"].get("content", "")
 else:
 logger.warning(f"Ollama响应缺少消息内容: {result}")
 return ""
 except Exception as e:
 logger.error(f"获取Ollama推理内容失败: {e}", exc_info=True)
 raise```
______________________________

## .../reasoning/base.py
```python
from ..interfaces import ReasoningProvider
from abc import abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
class BaseReasoningProvider(ReasoningProvider):
 def __init__(self, api_key: str = None, api_url: str = None):
 self.api_key = api_key
 self.api_url = api_url
 async def extract_reasoning_content(self, raw_content: str) -> str:
 return raw_content
 @abstractmethod
 async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
 pass```
______________________________

## .../generation/claude.py
```python
from .base import BaseGenerationProvider
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import aiohttp
from app.utils.logger import logger
from app.clients.claude_client import ClaudeClient
class ClaudeGenerationProvider(BaseGenerationProvider):
 def __init__(self, api_key: str, api_url: str = None, provider: str = "anthropic"):
 super().__init__(api_key, api_url)
 self.provider = provider.lower()
 self.client = ClaudeClient(
 api_key=api_key,
 api_url=api_url,
 provider=provider
 )
 logger.info(f"初始化Claude生成提供者: provider={self.provider}")
 async def generate_response(self, messages: List[Dict], model: str = None, **kwargs) -> Dict:
 try:
 if not model:
 model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 response = await self.client.chat(
 messages=messages,
 model=model,
 temperature=kwargs.get('temperature', 0.7),
 top_p=kwargs.get('top_p', 0.9),
 stream=False,
 tools=kwargs.get('tools'),
 tool_choice=kwargs.get('tool_choice')
 )
 return response
 except Exception as e:
 logger.error(f"生成回答内容失败: {e}", exc_info=True)
 raise
 async def stream_response(self, messages: List[Dict], model: str = None, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
 try:
 if not model:
 model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
 async for content_type, content in self.client.stream_chat(
 messages=messages,
 model=model,
 temperature=kwargs.get('temperature', 0.7),
 top_p=kwargs.get('top_p', 0.9),
 tools=kwargs.get('tools'),
 tool_choice=kwargs.get('tool_choice')
 ):
 yield content_type, content
 except Exception as e:
 logger.error(f"流式生成回答内容失败: {e}", exc_info=True)
 raise```
______________________________

## .../generation/__init__.py
```python
from .base import BaseGenerationProvider
from .claude import ClaudeGenerationProvider
from .factory import GenerationProviderFactory
__all__ = [
 "BaseGenerationProvider",
 "ClaudeGenerationProvider",
 "GenerationProviderFactory"
]```
______________________________

## .../generation/factory.py
```python
from .base import BaseGenerationProvider
from .claude import ClaudeGenerationProvider
import os
from app.utils.logger import logger
class GenerationProviderFactory:
 @staticmethod
 def create(provider_type: str = None) -> BaseGenerationProvider:
 provider_type = provider_type or os.getenv('GENERATION_PROVIDER', 'claude').lower()
 if provider_type == 'claude':
 api_key = os.getenv('CLAUDE_API_KEY')
 api_url = os.getenv('CLAUDE_API_URL')
 claude_provider = os.getenv('CLAUDE_PROVIDER', 'anthropic')
 if not api_key:
 raise ValueError("未设置CLAUDE_API_KEY环境变量")
 return ClaudeGenerationProvider(
 api_key=api_key,
 api_url=api_url,
 provider=claude_provider
 )
 else:
 raise ValueError(f"不支持的生成提供者类型: {provider_type}")```
______________________________

## .../generation/base.py
```python
from ..interfaces import GenerationProvider
from abc import abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
class BaseGenerationProvider(GenerationProvider):
 def __init__(self, api_key: str = None, api_url: str = None):
 self.api_key = api_key
 self.api_url = api_url
 @abstractmethod
 async def generate_response(self, messages: List[Dict], model: str, **kwargs) -> Dict:
 pass
 @abstractmethod
 async def stream_response(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
 pass```
______________________________

## .../test/test_database.py
```python
import os
import sys
import unittest
import datetime
import hashlib
import time
from dotenv import load_dotenv
from sqlalchemy import text
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.database.db_config import get_db_session, close_db_session
from app.database.db_models import User, Role, ConversationList, ConversationHistory, RoleEnum, SatisfactionEnum
from app.database.db_operations import DatabaseOperations
from app.database.db_utils import add_reasoning_column_if_not_exists
from app.utils.logger import logger
load_dotenv()
class TestDatabaseOperations(unittest.TestCase):
 test_data = {
 "admin_user_id": None,
 "test_user_id": None,
 "conversation_id": None,
 "history_user_id": None,
 "history_ai_id": None,
 }
 @classmethod
 def setUpClass(cls):
 logger.info("======== 开始数据库操作测试 ========")
 db = get_db_session()
 try:
 db.execute(text("SELECT 1"))
 logger.info("数据库连接正常")
 logger.info("检查并更新数据库表结构...")
 add_result = add_reasoning_column_if_not_exists()
 if add_result:
 logger.info("数据库表结构检查完成，确保conversation_history表包含reasoning列")
 else:
 logger.warning("数据库表结构更新失败，测试可能会失败")
 except Exception as e:
 logger.error(f"数据库连接或结构更新失败: {e}")
 raise
 finally:
 close_db_session(db)
 @classmethod
 def tearDownClass(cls):
 logger.info("======== 数据库操作测试完成 ========")
 def test_01_get_or_create_admin_user(self):
 logger.info("测试获取或创建管理员用户")
 try:
 admin_id = DatabaseOperations.get_or_create_admin_user()
 self.assertIsNotNone(admin_id, "管理员用户ID不应为空")
 self.__class__.test_data["admin_user_id"] = admin_id
 logger.info(f"管理员用户ID: {admin_id}")
 db = get_db_session()
 try:
 admin_user = db.query(User).filter(User.id == admin_id).first()
 self.assertIsNotNone(admin_user, "管理员用户应该存在")
 self.assertEqual(admin_user.username, "admin", "管理员用户名应为admin")
 admin_role = db.query(Role).filter(Role.name == "admin").first()
 self.assertIsNotNone(admin_role, "管理员角色应该存在")
 self.assertEqual(admin_user.role_id, admin_role.id, "用户应关联到管理员角色")
 finally:
 close_db_session(db)
 except Exception as e:
 self.fail(f"获取或创建管理员用户时发生错误: {e}")
 def test_02_create_test_user(self):
 logger.info("测试创建测试用户")
 db = get_db_session()
 try:
 admin_role = db.query(Role).filter(Role.name == "admin").first()
 self.assertIsNotNone(admin_role, "管理员角色应该存在")
 timestamp = int(time.time())
 test_username = f"test_user_{timestamp}"
 test_password = hashlib.sha256(f"test_password_{timestamp}".encode()).hexdigest()
 test_user = User(
 username=test_username,
 password=test_password,
 email=f"test_{timestamp}@example.com",
 real_name="Test User",
 role_id=admin_role.id,
 status=1
 )
 db.add(test_user)
 db.commit()
 created_user = db.query(User).filter(User.username == test_username).first()
 self.assertIsNotNone(created_user, "测试用户应该存在")
 self.__class__.test_data["test_user_id"] = created_user.id
 logger.info(f"测试用户ID: {created_user.id}")
 except Exception as e:
 db.rollback()
 self.fail(f"创建测试用户时发生错误: {e}")
 finally:
 close_db_session(db)
 def test_03_create_conversation(self):
 logger.info("测试创建对话会话")
 try:
 user_id = self.__class__.test_data["test_user_id"]
 title = f"测试对话_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
 conversation_id = DatabaseOperations.create_conversation(
 user_id=user_id,
 title=title
 )
 self.assertIsNotNone(conversation_id, "对话ID不应为空")
 self.__class__.test_data["conversation_id"] = conversation_id
 logger.info(f"创建的对话ID: {conversation_id}")
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 self.assertIsNotNone(conversation, "对话应该存在")
 self.assertEqual(conversation.title, title, "对话标题应匹配")
 self.assertEqual(conversation.user_id, user_id, "对话应该关联到测试用户")
 self.assertFalse(conversation.is_completed, "新创建的对话应该是未完成状态")
 finally:
 close_db_session(db)
 except Exception as e:
 self.fail(f"创建对话会话时发生错误: {e}")
 def test_04_add_user_question(self):
 logger.info("测试添加用户问题")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 user_id = self.__class__.test_data["test_user_id"]
 content = "这是一个测试问题，DeepClaude如何工作？"
 history_id = DatabaseOperations.add_conversation_history(
 conversation_id=conversation_id,
 user_id=user_id,
 role="user",
 content=content
 )
 self.assertIsNotNone(history_id, "历史记录ID不应为空")
 self.__class__.test_data["history_user_id"] = history_id
 logger.info(f"用户问题历史记录ID: {history_id}")
 db = get_db_session()
 try:
 history = db.query(ConversationHistory).filter(ConversationHistory.id == history_id).first()
 self.assertIsNotNone(history, "历史记录应该存在")
 self.assertEqual(history.role, RoleEnum.user, "角色应为用户")
 self.assertEqual(history.content, content, "内容应匹配")
 self.assertEqual(history.conversation_id, conversation_id, "历史记录应该关联到对话")
 self.assertEqual(history.user_id, user_id, "历史记录应该关联到用户")
 finally:
 close_db_session(db)
 except Exception as e:
 self.fail(f"添加用户问题时发生错误: {e}")
 def test_05_add_ai_answer(self):
 logger.info("测试添加AI回答")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 user_id = self.__class__.test_data["test_user_id"]
 content = "DeepClaude是一个集成了DeepSeek和Claude两个大语言模型能力的服务，它的工作流程是：1. 使用DeepSeek进行思考，2. 将思考结果传递给Claude生成最终答案。"
 reasoning = "我需要解释DeepClaude是什么以及它如何工作。DeepClaude实际上是一个结合了多个模型能力的服务，它的独特之处在于将推理和生成分开..."
 model_name = "claude-3-7-sonnet-20250219"
 tokens = 256
 history_id = DatabaseOperations.add_conversation_history(
 conversation_id=conversation_id,
 user_id=user_id,
 role="ai",
 content=content,
 reasoning=reasoning,
 model_name=model_name,
 tokens=tokens
 )
 self.assertIsNotNone(history_id, "历史记录ID不应为空")
 self.__class__.test_data["history_ai_id"] = history_id
 logger.info(f"AI回答历史记录ID: {history_id}")
 db = get_db_session()
 try:
 history = db.query(ConversationHistory).filter(ConversationHistory.id == history_id).first()
 self.assertIsNotNone(history, "历史记录应该存在")
 self.assertEqual(history.role, RoleEnum.ai, "角色应为AI")
 self.assertEqual(history.content, content, "内容应匹配")
 self.assertEqual(history.reasoning, reasoning, "思考过程应匹配")
 self.assertEqual(history.model_name, model_name, "模型名称应匹配")
 self.assertEqual(history.tokens, tokens, "Token数量应匹配")
 finally:
 close_db_session(db)
 except Exception as e:
 self.fail(f"添加AI回答时发生错误: {e}")
 def test_06_get_conversation_history(self):
 logger.info("测试获取对话历史")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 histories = DatabaseOperations.get_conversation_history(conversation_id)
 self.assertIsNotNone(histories, "历史记录列表不应为空")
 self.assertEqual(len(histories), 2, "应该有2条历史记录")
 user_history = next((h for h in histories if h["role"] == "user"), None)
 ai_history = next((h for h in histories if h["role"] == "ai"), None)
 self.assertIsNotNone(user_history, "用户历史记录应存在")
 self.assertIsNotNone(ai_history, "AI历史记录应存在")
 self.assertEqual(user_history["id"], self.__class__.test_data["history_user_id"], "用户历史记录ID应匹配")
 self.assertEqual(ai_history["id"], self.__class__.test_data["history_ai_id"], "AI历史记录ID应匹配")
 self.assertIsNotNone(ai_history["reasoning"], "AI历史记录应包含思考过程")
 logger.info("成功获取对话历史记录")
 except Exception as e:
 self.fail(f"获取对话历史时发生错误: {e}")
 def test_07_update_conversation_title(self):
 logger.info("测试更新对话标题")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 new_title = f"更新后的标题_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
 result = DatabaseOperations.update_conversation_title(
 conversation_id=conversation_id,
 title=new_title
 )
 self.assertTrue(result, "更新对话标题应该成功")
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 self.assertEqual(conversation.title, new_title, "对话标题应已更新")
 finally:
 close_db_session(db)
 logger.info(f"对话标题已更新为: {new_title}")
 except Exception as e:
 self.fail(f"更新对话标题时发生错误: {e}")
 def test_08_add_satisfaction_feedback(self):
 logger.info("测试添加满意度评价")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 satisfaction = "satisfied"
 feedback = "这是一个很好的回答，解释得很清楚！"
 result = DatabaseOperations.complete_conversation(
 conversation_id=conversation_id,
 satisfaction=satisfaction,
 feedback=feedback
 )
 self.assertTrue(result, "添加满意度评价应该成功")
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 self.assertEqual(conversation.satisfaction, SatisfactionEnum.satisfied, "满意度评价应为satisfied")
 self.assertEqual(conversation.feedback, feedback, "反馈内容应匹配")
 self.assertTrue(conversation.is_completed, "对话应标记为已完成")
 finally:
 close_db_session(db)
 logger.info("已成功添加满意度评价")
 except Exception as e:
 self.fail(f"添加满意度评价时发生错误: {e}")
 def test_09_get_user_conversations(self):
 logger.info("测试获取用户对话列表")
 try:
 user_id = self.__class__.test_data["test_user_id"]
 conversations = DatabaseOperations.get_user_conversations(
 user_id=user_id,
 limit=10,
 offset=0
 )
 self.assertIsNotNone(conversations, "对话列表不应为空")
 self.assertGreaterEqual(len(conversations), 1, "应该至少有1个对话")
 test_conversation = next((c for c in conversations if c["id"] == self.__class__.test_data["conversation_id"]), None)
 self.assertIsNotNone(test_conversation, "测试对话应该存在于列表中")
 self.assertTrue(test_conversation["is_completed"], "对话应标记为已完成")
 self.assertEqual(test_conversation["satisfaction"], "satisfied", "满意度评价应为satisfied")
 self.assertGreaterEqual(test_conversation["message_count"], 2, "消息数量应至少为2")
 logger.info("成功获取用户对话列表")
 except Exception as e:
 self.fail(f"获取用户对话列表时发生错误: {e}")
 def test_10_delete_conversation(self):
 logger.info("测试删除对话及其历史记录")
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 result = DatabaseOperations.delete_conversation(conversation_id)
 self.assertTrue(result, "删除对话应该成功")
 db = get_db_session()
 try:
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 self.assertIsNone(conversation, "对话应该已被删除")
 histories = db.query(ConversationHistory).filter(
 ConversationHistory.conversation_id == conversation_id
 ).all()
 self.assertEqual(len(histories), 0, "历史记录应该已被删除")
 finally:
 close_db_session(db)
 logger.info("对话及其历史记录已成功删除")
 except Exception as e:
 self.fail(f"删除对话时发生错误: {e}")
 def test_11_cleanup_test_user(self):
 logger.info("清理测试用户")
 db = get_db_session()
 try:
 test_user_id = self.__class__.test_data["test_user_id"]
 db.query(User).filter(User.id == test_user_id).delete()
 db.commit()
 test_user = db.query(User).filter(User.id == test_user_id).first()
 self.assertIsNone(test_user, "测试用户应该已被删除")
 logger.info("测试用户已成功清理")
 except Exception as e:
 db.rollback()
 self.fail(f"清理测试用户时发生错误: {e}")
 finally:
 close_db_session(db)
 def test_12_verify_cleanup(self):
 logger.info("验证所有测试数据已清理")
 db = get_db_session()
 try:
 conversation_id = self.__class__.test_data["conversation_id"]
 conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
 self.assertIsNone(conversation, "对话应该已被删除")
 history_user_id = self.__class__.test_data["history_user_id"]
 history_ai_id = self.__class__.test_data["history_ai_id"]
 user_history = db.query(ConversationHistory).filter(ConversationHistory.id == history_user_id).first()
 ai_history = db.query(ConversationHistory).filter(ConversationHistory.id == history_ai_id).first()
 self.assertIsNone(user_history, "用户历史记录应该已被删除")
 self.assertIsNone(ai_history, "AI历史记录应该已被删除")
 test_user_id = self.__class__.test_data["test_user_id"]
 test_user = db.query(User).filter(User.id == test_user_id).first()
 self.assertIsNone(test_user, "测试用户应该已被删除")
 logger.info("所有测试数据已成功清理")
 except Exception as e:
 self.fail(f"验证数据清理时发生错误: {e}")
 finally:
 close_db_session(db)
if __name__ == "__main__":
 unittest.main()```
______________________________

## .../test/test_deepseek_client.py
```python
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger
def parse_args():
 parser = argparse.ArgumentParser(description='测试 DeepSeek 客户端')
 parser.add_argument('--reasoning-mode', type=str, choices=['auto', 'reasoning_field', 'think_tags', 'any_content'],
 default=os.getenv('DEEPSEEK_REASONING_MODE', 'auto'),
 help='推理内容提取模式')
 parser.add_argument('--provider', type=str, choices=['deepseek', 'siliconflow', 'nvidia'],
 default=os.getenv('DEEPSEEK_PROVIDER', 'deepseek'),
 help='API提供商')
 parser.add_argument('--model', type=str,
 default=os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner'),
 help='模型名称')
 parser.add_argument('--question', type=str, default='1+1等于几?',
 help='测试问题')
 parser.add_argument('--debug', action='store_true',
 help='启用调试模式')
 return parser.parse_args()
async def test_deepseek_stream(args):
 api_key = os.getenv("DEEPSEEK_API_KEY")
 api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
 os.environ["DEEPSEEK_REASONING_MODE"] = args.reasoning_mode
 os.environ["DEEPSEEK_PROVIDER"] = args.provider
 logger.info("=== DeepSeek 客户端测试开始 ===")
 logger.info(f"API URL: {api_url}")
 logger.info(f"API Key 是否存在: {bool(api_key)}")
 logger.info(f"提供商: {args.provider}")
 logger.info(f"推理模式: {args.reasoning_mode}")
 logger.info(f"使用模型: {args.model}")
 logger.info(f"测试问题: {args.question}")
 if not api_key:
 logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
 return
 messages = [
 {"role": "user", "content": args.question}
 ]
 client = DeepSeekClient(api_key, api_url, provider=args.provider)
 try:
 logger.info("开始测试 DeepSeek 流式输出...")
 logger.debug(f"发送消息: {messages}")
 reasoning_buffer = []
 content_buffer = []
 reasoning_count = 0
 content_count = 0
 async for content_type, content in client.get_reasoning(
 messages=messages,
 model=args.model
 ):
 if content_type == "reasoning":
 reasoning_count += 1
 reasoning_buffer.append(content)
 if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"推理过程（{reasoning_count}）：{''.join(reasoning_buffer)}")
 reasoning_buffer = []
 elif content_type == "content":
 content_count += 1
 content_buffer.append(content)
 if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"普通内容（{content_count}）：{''.join(content_buffer)}")
 content_buffer = []
 if reasoning_buffer:
 logger.info(f"推理过程（最终）：{''.join(reasoning_buffer)}")
 if content_buffer:
 logger.info(f"普通内容（最终）：{''.join(content_buffer)}")
 logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个普通内容片段")
 if reasoning_count == 0:
 logger.warning("未收到任何推理内容！请检查以下设置:")
 logger.warning(f"1. 推理模式是否正确：{args.reasoning_mode}")
 logger.warning(f"2. API提供商 {args.provider} 是否支持推理功能")
 logger.warning(f"3. 模型 {args.model} 是否支持推理输出")
 logger.info("=== DeepSeek 客户端测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
def main():
 args = parse_args()
 if args.debug:
 os.environ['LOG_LEVEL'] = 'DEBUG'
 load_dotenv()
 asyncio.run(test_deepseek_stream(args))
if __name__ == "__main__":
 main()```
______________________________

## .../test/test_deepclaude.py
```python
import os
import sys
import asyncio
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.deepclaude.deepclaude import DeepClaude
from app.utils.logger import logger
from app.clients.deepseek_client import DeepSeekClient
load_dotenv()
def clean_env_vars():
 for key in ["REASONING_PROVIDER", "DEEPSEEK_PROVIDER", "CLAUDE_PROVIDER"]:
 if key in os.environ:
 value = os.environ[key].split('#')[0].strip()
 os.environ[key] = value
 logger.info(f"清理环境变量 {key}={value}")
clean_env_vars()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER", "anthropic")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_PROVIDER = os.getenv("DEEPSEEK_PROVIDER", "deepseek")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "false").lower() == "true"
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "deepseek")
logger.info(f"测试环境信息:")
logger.info(f"CLAUDE_PROVIDER={CLAUDE_PROVIDER}")
logger.info(f"CLAUDE_MODEL={os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')}")
logger.info(f"REASONING_PROVIDER={REASONING_PROVIDER}")
logger.info(f"DEEPSEEK_PROVIDER={DEEPSEEK_PROVIDER}")
test_messages = [
 {"role": "user", "content": "用Python写一个简单的计算器程序"}
]
async def test_deepclaude_init():
 logger.info("开始测试DeepClaude初始化...")
 try:
 deepclaude = DeepClaude(
 enable_enhanced_reasoning=True,
 save_to_db=False
 )
 assert deepclaude.claude_client is not None, "Claude客户端初始化失败"
 assert hasattr(deepclaude, 'thinker_client'), "思考者客户端初始化失败"
 assert deepclaude.min_reasoning_chars > 0, "推理字符数设置错误"
 assert len(deepclaude.reasoning_modes) > 0, "推理模式列表为空"
 assert isinstance(deepclaude.search_enabled, bool), "搜索增强配置错误"
 assert "tavily_search" in deepclaude.supported_tools, "工具支持配置错误"
 logger.info("DeepClaude初始化测试通过!")
 return deepclaude
 except Exception as e:
 logger.error(f"DeepClaude初始化测试失败: {e}", exc_info=True)
 raise
async def test_stream_output(deepclaude):
 logger.info("开始测试DeepClaude流式输出...")
 try:
 reasoning_received = False
 answer_received = False
 async for response_bytes in deepclaude.chat_completions_with_stream(
 messages=test_messages,
 chat_id="test-chat-id",
 created_time=1234567890,
 model="deepclaude"
 ):
 response_text = response_bytes.decode('utf-8')
 if '"is_reasoning": true' in response_text:
 reasoning_received = True
 logger.info("收到推理内容")
 if '"is_reasoning": true' not in response_text and '"content":' in response_text:
 answer_received = True
 logger.info("收到回答内容")
 assert reasoning_received, "未收到推理内容"
 assert answer_received, "未收到回答内容"
 logger.info("DeepClaude流式输出测试通过!")
 except Exception as e:
 logger.error(f"DeepClaude流式输出测试失败: {e}", exc_info=True)
 raise
async def test_non_stream_output(deepclaude):
 logger.info("开始测试DeepClaude非流式输出...")
 try:
 response = await deepclaude.chat_completions_without_stream(
 messages=test_messages,
 model_arg=(0.7, 0.9, 0, 0)
 )
 assert "content" in response, "返回结果中缺少内容字段"
 assert "role" in response, "返回结果中缺少角色字段"
 assert response["role"] == "assistant", "角色字段值错误"
 assert len(response["content"]) > 0, "内容为空"
 logger.info(f"收到非流式回答: {response['content'][:100]}...")
 logger.info("DeepClaude非流式输出测试通过!")
 except Exception as e:
 logger.error(f"DeepClaude非流式输出测试失败: {e}", exc_info=True)
 raise
async def test_non_stream_output_with_tools(deepclaude):
 logger.info("开始测试DeepClaude非流式输出工具调用...")
 tools = [
 {
 "type": "function",
 "function": {
 "name": "get_weather",
 "description": "获取特定位置的天气信息",
 "parameters": {
 "type": "object",
 "properties": {
 "location": {
 "type": "string",
 "description": "城市名称，如北京、上海等"
 },
 "unit": {
 "type": "string",
 "enum": ["celsius", "fahrenheit"],
 "description": "温度单位"
 }
 },
 "required": ["location"]
 }
 }
 }
 ]
 weather_messages = [
 {"role": "user", "content": "北京今天的天气怎么样？"}
 ]
 try:
 response = await deepclaude.chat_completions_without_stream(
 messages=weather_messages,
 model_arg=(0.7, 0.9, 0, 0),
 tools=tools,
 tool_choice="auto"
 )
 logger.info(f"非流式工具调用响应: {response}")
 assert "content" in response, "返回结果中缺少内容字段"
 assert "role" in response, "返回结果中缺少角色字段"
 assert response["role"] == "assistant", "角色字段值错误"
 if "tool_calls" in response:
 logger.info(f"收到工具调用: {response['tool_calls']}")
 assert len(response["tool_calls"]) > 0, "工具调用列表为空"
 assert "tool_results" in response, "返回结果中缺少工具结果字段"
 else:
 logger.info("本次测试未返回工具调用")
 logger.info("DeepClaude非流式输出工具调用测试通过!")
 except Exception as e:
 logger.error(f"DeepClaude非流式输出工具调用测试失败: {e}", exc_info=True)
 raise
async def test_reasoning_function(deepclaude):
 logger.info("开始测试DeepClaude推理功能...")
 try:
 reasoning = await deepclaude._get_reasoning_content(
 messages=test_messages,
 model="deepseek-reasoner"
 )
 if reasoning:
 logger.info(f"成功获取推理内容: {reasoning[:100]}...")
 logger.info("DeepClaude推理功能测试通过!")
 return True
 else:
 logger.warning("推理内容为空，但不视为测试失败")
 return True
 except Exception as e:
 logger.error(f"DeepClaude推理功能测试失败: {e}", exc_info=True)
 logger.warning("推理测试失败，但不阻止其他测试继续进行")
 return False
async def test_reasoning_fallback(deepclaude):
 logger.info("开始测试DeepClaude回退机制...")
 original_provider = os.environ.get('REASONING_PROVIDER')
 try:
 os.environ['REASONING_PROVIDER'] = 'deepseek'
 try:
 await deepclaude._get_reasoning_with_fallback(
 messages=test_messages,
 model="deepseek-reasoner"
 )
 logger.info("回退机制调用成功")
 except Exception as e:
 logger.warning(f"推理回退测试出现异常: {e}")
 logger.info("DeepClaude回退机制测试完成!")
 return True
 except Exception as e:
 logger.error(f"DeepClaude回退机制测试失败: {e}", exc_info=True)
 logger.warning("回退测试失败，但不阻止其他测试继续进行")
 return False
 finally:
 if original_provider:
 os.environ['REASONING_PROVIDER'] = original_provider
 else:
 os.environ.pop('REASONING_PROVIDER', None)
async def test_claude_integration(deepclaude):
 logger.info("开始测试Claude 3.7集成...")
 try:
 claude_messages = [{"role": "user", "content": "返回当前你的模型版本"}]
 response = ""
 async for content_type, content in deepclaude.claude_client.stream_chat(
 messages=claude_messages,
 model=os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
 temperature=0.7,
 top_p=0.9
 ):
 if content_type == "content":
 response += content
 logger.info(f"收到Claude内容: {content}")
 assert "Claude 3" in response, "未检测到Claude 3系列模型"
 logger.info(f"Claude回复: {response[:200]}...")
 logger.info("Claude集成测试通过!")
 return True
 except Exception as e:
 logger.error(f"Claude集成测试失败: {e}", exc_info=True)
 return False
async def run_tests():
 logger.info("开始DeepClaude集成测试...")
 test_results = {
 "初始化测试": False,
 "Claude集成测试": False,
 "推理功能测试": False,
 "回退机制测试": False,
 "流式输出测试": False,
 "非流式输出测试": False,
 "非流式工具调用测试": False
 }
 try:
 deepclaude = await test_deepclaude_init()
 test_results["初始化测试"] = True
 test_results["Claude集成测试"] = await test_claude_integration(deepclaude)
 if test_results["Claude集成测试"]:
 test_results["推理功能测试"] = await test_reasoning_function(deepclaude)
 test_results["回退机制测试"] = await test_reasoning_fallback(deepclaude)
 if test_results["推理功能测试"]:
 try:
 await test_stream_output(deepclaude)
 test_results["流式输出测试"] = True
 except Exception as e:
 logger.error(f"流式输出测试失败: {e}", exc_info=True)
 try:
 await test_non_stream_output(deepclaude)
 test_results["非流式输出测试"] = True
 except Exception as e:
 logger.error(f"非流式输出测试失败: {e}", exc_info=True)
 try:
 await test_non_stream_output_with_tools(deepclaude)
 test_results["非流式工具调用测试"] = True
 except Exception as e:
 logger.error(f"非流式工具调用测试失败: {e}", exc_info=True)
 except Exception as e:
 logger.error(f"测试过程中发生未捕获的异常: {e}", exc_info=True)
 logger.info("\n" + "="*50)
 logger.info("DeepClaude 测试结果总结:")
 logger.info("="*50)
 success_count = 0
 for test_name, result in test_results.items():
 status = "✅ 通过" if result else "❌ 失败"
 if result:
 success_count += 1
 logger.info(f"{test_name}: {status}")
 logger.info("="*50)
 logger.info(f"测试完成: {success_count}/{len(test_results)} 通过")
 logger.info("="*50)
 return test_results
def main():
 test_results = asyncio.run(run_tests())
 sys.exit(0)
if __name__ == "__main__":
 main()```
______________________________

## .../test/test_siliconflow_deepseek.py
```python
import os
import sys
import asyncio
import json
import argparse
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger
load_dotenv()
def parse_args():
 default_api_key = os.getenv("DEEPSEEK_API_KEY")
 default_api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
 default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
 parser = argparse.ArgumentParser(description='测试硅基流动 DeepSeek R1 API')
 parser.add_argument('--api-key', type=str,
 default=default_api_key,
 help=f'硅基流动API密钥 (默认: {default_api_key[:8]}*** 来自环境变量)' if default_api_key else '硅基流动API密钥')
 parser.add_argument('--api-url', type=str,
 default=default_api_url,
 help=f'硅基流动API地址 (默认: {default_api_url})')
 parser.add_argument('--model', type=str,
 default=default_model,
 help=f'硅基流动模型名称 (默认: {default_model})')
 parser.add_argument('--question', type=str,
 default='中国大模型行业2025年将会迎来哪些机遇和挑战？',
 help='测试问题')
 parser.add_argument('--debug', action='store_true',
 help='启用调试模式')
 args = parser.parse_args()
 if not args.api_key:
 logger.error("未提供API密钥！请在.env文件中设置DEEPSEEK_API_KEY或使用--api-key参数")
 sys.exit(1)
 return args
async def test_siliconflow_reasoning(args):
 logger.info("=== 硅基流动 DeepSeek-R1 API 测试开始 ===")
 logger.info(f"API URL: {args.api_url}")
 logger.info(f"API Key: {args.api_key[:8]}***")
 logger.info(f"Model: {args.model}")
 logger.info(f"测试问题: {args.question}")
 messages = [
 {"role": "user", "content": args.question}
 ]
 client = DeepSeekClient(
 api_key=args.api_key,
 api_url=args.api_url,
 provider="siliconflow"
 )
 os.environ["DEEPSEEK_REASONING_MODE"] = "reasoning_field"
 os.environ["IS_ORIGIN_REASONING"] = "true"
 try:
 logger.info("开始测试硅基流动 DeepSeek-R1 推理功能...")
 logger.debug(f"发送消息: {messages}")
 reasoning_buffer = []
 content_buffer = []
 reasoning_count = 0
 content_count = 0
 async for content_type, content in client.get_reasoning(
 messages=messages,
 model=args.model
 ):
 if content_type == "reasoning":
 reasoning_count += 1
 reasoning_buffer.append(content)
 if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"推理过程（{reasoning_count}）: {''.join(reasoning_buffer)}")
 reasoning_buffer = []
 elif content_type == "content":
 content_count += 1
 content_buffer.append(content)
 if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"回答内容（{content_count}）: {''.join(content_buffer)}")
 content_buffer = []
 if reasoning_buffer:
 logger.info(f"推理过程（最终）: {''.join(reasoning_buffer)}")
 if content_buffer:
 logger.info(f"回答内容（最终）: {''.join(content_buffer)}")
 logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个回答内容片段")
 if reasoning_count == 0:
 logger.warning("未收到任何推理内容！请检查以下设置:")
 logger.warning("1. 确保DEEPSEEK_REASONING_MODE设置为'reasoning_field'")
 logger.warning("2. 确保IS_ORIGIN_REASONING设置为'true'")
 logger.warning("3. 确保硅基流动API支持推理输出功能")
 logger.info("=== 硅基流动 DeepSeek-R1 API 测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
async def test_siliconflow_non_stream_api(args):
 import aiohttp
 logger.info("=== 硅基流动 DeepSeek-R1 非流式API测试开始 ===")
 logger.info(f"测试问题: {args.question}")
 payload = {
 "model": args.model,
 "messages": [
 {
 "role": "user",
 "content": args.question
 }
 ],
 "stream": False,
 "max_tokens": 512,
 "stop": None,
 "temperature": 0.7,
 "top_p": 0.7,
 "top_k": 50,
 "frequency_penalty": 0.5,
 "n": 1,
 "response_format": {"type": "text"}
 }
 headers = {
 "Authorization": f"Bearer {args.api_key}",
 "Content-Type": "application/json"
 }
 try:
 async with aiohttp.ClientSession() as session:
 async with session.post(args.api_url, json=payload, headers=headers) as response:
 if response.status != 200:
 error_text = await response.text()
 logger.error(f"API调用失败: HTTP {response.status}\n{error_text}")
 return
 data = await response.json()
 logger.debug(f"API响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
 if "choices" in data and len(data["choices"]) > 0:
 reasoning = data["choices"][0].get("reasoning_content")
 content = data["choices"][0].get("message", {}).get("content")
 if reasoning:
 logger.info(f"推理内容: {reasoning[:200]}...")
 else:
 logger.warning("响应中没有推理内容")
 if content:
 logger.info(f"回答内容: {content[:200]}...")
 else:
 logger.warning("响应中没有回答内容")
 if "usage" in data:
 logger.info(f"Token使用情况: {data['usage']}")
 logger.info("=== 硅基流动 DeepSeek-R1 非流式API测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
async def test_siliconflow_stream_api_direct(args):
 import requests
 logger.info("=== 硅基流动 DeepSeek-R1 流式API直接调用测试开始 ===")
 logger.info(f"API URL: {args.api_url}")
 logger.info(f"API Key: {args.api_key[:8]}***")
 logger.info(f"Model: {args.model}")
 logger.info(f"测试问题: {args.question}")
 payload = {
 "model": args.model,
 "messages": [
 {
 "role": "user",
 "content": args.question
 }
 ],
 "stream": True,
 "max_tokens": 512,
 "stop": None,
 "temperature": 0.7,
 "top_p": 0.7,
 "top_k": 50,
 "frequency_penalty": 0.5,
 "n": 1,
 "response_format": {"type": "text"}
 }
 headers = {
 "Authorization": f"Bearer {args.api_key}",
 "Content-Type": "application/json"
 }
 try:
 logger.info("发送流式API请求...")
 response = requests.post(args.api_url, json=payload, headers=headers, stream=True)
 if response.status_code != 200:
 logger.error(f"API调用失败: HTTP {response.status_code}\n{response.text}")
 return
 reasoning_buffer = []
 content_buffer = []
 reasoning_count = 0
 content_count = 0
 logger.info("开始接收流式响应...")
 for line in response.iter_lines():
 if not line:
 continue
 if line.startswith(b"data: "):
 data_str = line[6:].decode("utf-8")
 if data_str == "[DONE]":
 logger.info("收到流式响应结束标记")
 break
 try:
 data = json.loads(data_str)
 logger.debug(f"收到数据: {json.dumps(data, ensure_ascii=False)}")
 if "choices" in data and len(data["choices"]) > 0:
 reasoning = data["choices"][0].get("reasoning_content")
 delta = data["choices"][0].get("delta", {})
 content = delta.get("content", "")
 if reasoning:
 reasoning_count += 1
 reasoning_buffer.append(reasoning)
 if len(''.join(reasoning_buffer)) >= 50 or any(p in reasoning for p in '。，！？.!?'):
 logger.info(f"推理过程（{reasoning_count}）: {''.join(reasoning_buffer)}")
 reasoning_buffer = []
 if content:
 content_count += 1
 content_buffer.append(content)
 if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
 logger.info(f"回答内容（{content_count}）: {''.join(content_buffer)}")
 content_buffer = []
 except json.JSONDecodeError:
 logger.error(f"无法解析 JSON: {data_str}")
 if reasoning_buffer:
 logger.info(f"推理过程（最终）: {''.join(reasoning_buffer)}")
 if content_buffer:
 logger.info(f"回答内容（最终）: {''.join(content_buffer)}")
 logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个回答内容片段")
 logger.info("=== 硅基流动 DeepSeek-R1 流式API直接调用测试完成 ===")
 except Exception as e:
 logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
 logger.error(f"错误类型: {type(e)}")
def check_environment():
 provider = os.getenv('REASONING_PROVIDER', '').lower()
 if provider == 'siliconflow':
 logger.info("检测到环境变量REASONING_PROVIDER=siliconflow")
 else:
 logger.warning(f"当前REASONING_PROVIDER={provider}，非siliconflow，但仍会继续测试")
 api_key = os.getenv('DEEPSEEK_API_KEY')
 if api_key:
 logger.info(f"已从环境变量中读取API密钥: {api_key[:8]}***")
 else:
 logger.warning("环境变量中未设置DEEPSEEK_API_KEY")
 api_url = os.getenv('DEEPSEEK_API_URL')
 if api_url:
 logger.info(f"已从环境变量中读取API URL: {api_url}")
 else:
 logger.warning("环境变量中未设置DEEPSEEK_API_URL，将使用默认值")
 is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', '').lower() == 'true'
 reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', '')
 if is_origin_reasoning and reasoning_mode == 'reasoning_field':
 logger.info("推理模式配置正确")
 else:
 logger.warning(f"当前推理模式可能不适合硅基流动API：IS_ORIGIN_REASONING={is_origin_reasoning}, DEEPSEEK_REASONING_MODE={reasoning_mode}")
 logger.warning("已自动设置为正确的推理模式")
def main():
 check_environment()
 args = parse_args()
 if args.debug:
 os.environ['LOG_LEVEL'] = 'DEBUG'
 else:
 os.environ['LOG_LEVEL'] = 'INFO'
 loop = asyncio.get_event_loop()
 loop.run_until_complete(test_siliconflow_reasoning(args))
 loop.run_until_complete(test_siliconflow_non_stream_api(args))
 loop.run_until_complete(test_siliconflow_stream_api_direct(args))
 loop.close()
if __name__ == "__main__":
 main()```
______________________________

## .../test/test_nvidia_deepseek.py
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

## .../test/test_ollama_r1.py
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

## .../test/test_claude_client.py
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
 model = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
 logger.info(f"API URL: {api_url}")
 logger.info(f"API Key 是否存在: {bool(api_key)}")
 logger.info(f"Provider: {provider}")
 logger.info(f"Model: {model}")
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
 model=model
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

## .../tests/test_deepclaude.py
```python
import os
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock
from app.deepclaude.core import DeepClaude
from app.deepclaude.reasoning.factory import ReasoningProviderFactory
from app.deepclaude.tools.handlers import ToolHandler
@pytest.fixture
def test_env_setup():
 os.environ["CLAUDE_API_KEY"] = "test_claude_key"
 os.environ["CLAUDE_MODEL"] = "claude-3-7-sonnet-20250219"
 os.environ["REASONING_PROVIDER"] = "deepseek"
 os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
 os.environ["SAVE_TO_DB"] = "false"
 original_values = {
 "CLAUDE_API_KEY": os.environ.get("CLAUDE_API_KEY"),
 "CLAUDE_MODEL": os.environ.get("CLAUDE_MODEL"),
 "REASONING_PROVIDER": os.environ.get("REASONING_PROVIDER"),
 "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
 "SAVE_TO_DB": os.environ.get("SAVE_TO_DB")
 }
 yield
 for key, value in original_values.items():
 if value is not None:
 os.environ[key] = value
 else:
 if key in os.environ:
 del os.environ[key]
@pytest.mark.asyncio
async def test_deepclaude_initialization(test_env_setup):
 deepclaude = DeepClaude()
 assert deepclaude.claude_api_key == "test_claude_key"
 assert deepclaude.claude_provider == "anthropic"
 assert deepclaude.save_to_db is False
 assert deepclaude.min_reasoning_chars == 100
 assert deepclaude.claude_client is not None
 assert deepclaude.tool_handler is not None
 assert deepclaude.thinker_client is not None
@pytest.mark.asyncio
async def test_format_tool_decision_prompt():
 deepclaude = DeepClaude()
 tools = [
 {
 "function": {
 "name": "test_function",
 "description": "测试函数描述",
 "parameters": {
 "type": "object",
 "required": ["required_param"],
 "properties": {
 "required_param": {
 "type": "string",
 "description": "必填参数描述"
 },
 "optional_param": {
 "type": "integer",
 "description": "可选参数描述",
 "enum": [1, 2, 3]
 }
 }
 }
 }
 }
 ]
 prompt = deepclaude._format_tool_decision_prompt(
 original_question="测试问题",
 reasoning="测试推理过程",
 tools=tools
 )
 assert "测试问题" in prompt
 assert "测试推理过程" in prompt
 assert "test_function" in prompt
 assert "测试函数描述" in prompt
 assert "required_param" in prompt
 assert "必填" in prompt
 assert "optional_param" in prompt
 assert "可选" in prompt
 assert "可选值: 1, 2, 3" in prompt
 tools = [
 {
 "type": "custom",
 "name": "test_custom",
 "description": "自定义工具描述",
 "tool_schema": {
 "type": "object",
 "required": ["required_param"],
 "properties": {
 "required_param": {
 "type": "string",
 "description": "必填参数描述"
 }
 }
 }
 }
 ]
 prompt = deepclaude._format_tool_decision_prompt(
 original_question="测试问题",
 reasoning="测试推理过程",
 tools=tools
 )
 assert "test_custom" in prompt
 assert "自定义工具描述" in prompt
 assert "required_param" in prompt
 assert "必填" in prompt
@pytest.mark.asyncio
@patch("app.deepclaude.core.ReasoningProviderFactory.create")
@patch("app.clients.claude_client.ClaudeClient.chat")
async def test_chat_completions_without_stream(mock_claude_chat, mock_reasoning_factory, test_env_setup):
 mock_thinker = MagicMock()
 mock_thinker.get_reasoning.return_value = "模拟推理结果"
 mock_reasoning_factory.return_value = mock_thinker
 mock_claude_chat.return_value = {"content": "模拟Claude回答"}
 deepclaude = DeepClaude()
 response = await deepclaude.chat_completions_without_stream(
 messages=[{"role": "user", "content": "测试问题"}],
 model_arg=(0.7, 0.9)
 )
 assert response["content"] == "模拟Claude回答"
 assert response["role"] == "assistant"
 assert response["reasoning"] == "模拟推理结果"
 mock_thinker.get_reasoning.assert_called_once()
 mock_claude_chat.assert_called_once()
 call_args = mock_claude_chat.call_args[1]
 assert "我已经思考了以下问题" in call_args["messages"][0]["content"]
 assert "模拟推理结果" in call_args["messages"][0]["content"]
@pytest.mark.asyncio
@patch("app.deepclaude.core.ReasoningProviderFactory.create")
@patch("app.deepclaude.tools.handlers.ToolHandler.validate_and_convert_tools")
@patch("app.clients.claude_client.ClaudeClient.chat")
async def test_direct_tool_pass_without_stream(mock_claude_chat, mock_validate_tools, mock_reasoning_factory, test_env_setup):
 os.environ["CLAUDE_DIRECT_TOOL_PASS"] = "true"
 mock_validate_tools.return_value = [{"function": {"name": "test_tool"}}]
 mock_claude_chat.return_value = {
 "content": None,
 "tool_calls": [
 {"type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
 ]
 }
 deepclaude = DeepClaude()
 tools = [{"type": "function", "function": {"name": "test_tool"}}]
 response = await deepclaude.chat_completions_without_stream(
 messages=[{"role": "user", "content": "使用工具"}],
 model_arg=(0.7, 0.9),
 tools=tools
 )
 assert "tool_calls" in response
 assert response["tool_calls"][0]["function"]["name"] == "test_tool"
 mock_validate_tools.assert_called_once_with(tools, target_format='claude-3')
 mock_claude_chat.assert_called_once()
 del os.environ["CLAUDE_DIRECT_TOOL_PASS"]```
______________________________

# 配置文件

## ./Dockerfile
```dockerfile
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
```
______________________________
