# DeepClaude项目错误修复与重构方案

## 一、当前错误分析

从错误信息来看，问题出在`deepclaude.py`文件的`_validate_and_convert_tools`方法中：

```
[2025-03-04 12:57:12] [ERROR] [deepclaude.py:434] 处理流式请求时发生错误: 'custom'
Traceback (most recent call last):
  File "/Users/wanglei/code/DeepClaude/app/deepclaude/deepclaude.py", line 316, in chat_completions_with_stream
    converted_tools = self._validate_and_convert_tools(tools, target_format='claude-3')
  File "/Users/wanglei/code/DeepClaude/app/deepclaude/deepclaude.py", line 1398, in _validate_and_convert_tools
    if tool["type"] == "custom" and isinstance(tool.get("custom", {}), dict) and "type" in tool["custom"]:
                                                                                           ~~~~^^^^^^^^^^
KeyError: 'custom'
```

**错误原因**：工具类型被标记为`"custom"`，但没有包含实际的`"custom"`字段数据。代码中假设`tool["type"] == "custom"`的工具一定会有`tool["custom"]`字段，然而这个假设是错误的。

## 二、即时错误修复方案

修改`_validate_and_convert_tools`方法中的条件判断，确保在检查`"custom"`字段内部属性前先确认该字段存在：

```python
# 修改前
if tool["type"] == "custom" and isinstance(tool.get("custom", {}), dict) and "type" in tool["custom"]:

# 修改后
if tool["type"] == "custom" and "custom" in tool and isinstance(tool["custom"], dict) and "type" in tool["custom"]:
```

或者更安全的写法：

```python
custom_field = tool.get("custom", {})
if tool["type"] == "custom" and isinstance(custom_field, dict) and "type" in custom_field:
```

## 三、系统重构方案

### 1. 总体架构设计

按照策略模式重构系统，将`deepclaude.py`文件拆分成多个模块，每个模块处理特定的功能：

```
app/
├── deepclaude/
│   ├── __init__.py
│   ├── core.py                  # 核心协调类
│   ├── interfaces.py            # 接口定义
│   ├── reasoning/               # 推理模块
│   │   ├── __init__.py
│   │   ├── base.py              # 推理基础接口
│   │   ├── deepseek.py          # DeepSeek推理实现
│   │   ├── ollama.py            # Ollama推理实现
│   │   └── factory.py           # 推理策略工厂
│   ├── generation/              # 生成模块
│   │   ├── __init__.py
│   │   ├── base.py              # 生成基础接口
│   │   ├── claude.py            # Claude生成实现
│   │   └── factory.py           # 生成策略工厂
│   ├── tools/                   # 工具模块
│   │   ├── __init__.py
│   │   ├── base.py              # 工具基础接口
│   │   ├── converters.py        # 工具格式转换
│   │   ├── handlers.py          # 工具处理实现
│   │   └── validators.py        # 工具验证逻辑
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── formatting.py        # 格式化工具
│       ├── prompts.py           # 提示词模板
│       └── streaming.py         # 流式处理工具
```

### 2. 接口定义

创建明确的接口定义，以保证各组件之间的一致性：

```python
# app/deepclaude/interfaces.py

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional

class ReasoningProvider(ABC):
    """推理服务提供者的接口定义"""
    
    @abstractmethod
    async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
        """获取推理内容"""
        pass

class GenerationProvider(ABC):
    """生成服务提供者的接口定义"""
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """生成回答内容"""
        pass
    
    @abstractmethod
    async def stream_response(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
        """流式生成回答内容"""
        pass

class ToolProcessor(ABC):
    """工具处理器的接口定义"""
    
    @abstractmethod
    def validate_and_convert(self, tools: List[Dict], target_format: str) -> List[Dict]:
        """验证并转换工具格式"""
        pass
    
    @abstractmethod
    async def process_tool_call(self, tool_call: Dict, **kwargs) -> Dict:
        """处理工具调用"""
        pass
```

### 3. 具体实现类

#### 推理模块

```python
# app/deepclaude/reasoning/base.py
from ..interfaces import ReasoningProvider
from abc import abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional

class BaseReasoningProvider(ReasoningProvider):
    """推理提供者基类"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key
        self.api_url = api_url
        
    async def extract_reasoning_content(self, raw_content: str) -> str:
        """提取推理内容的方法，可被子类重写"""
        return raw_content
        
    @abstractmethod
    async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
        """获取推理内容的抽象方法，必须由子类实现"""
        pass
```

```python
# app/deepclaude/reasoning/deepseek.py
from .base import BaseReasoningProvider
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import aiohttp
from app.utils.logger import logger

class DeepSeekReasoningProvider(BaseReasoningProvider):
    """基于DeepSeek的推理提供者"""
    
    def __init__(self, api_key: str, api_url: str, provider: str = "deepseek"):
        super().__init__(api_key, api_url)
        self.provider = provider
        self.reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto')
        
    async def extract_reasoning_from_think_tags(self, content: str) -> str:
        """从<think>标签中提取推理内容"""
        if "<think>" in content and "</think>" in content:
            start = content.find("<think>") + 7
            end = content.find("</think>")
            if start < end:
                return content[start:end].strip()
        return ""
        
    async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
        """获取DeepSeek推理内容"""
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
        
        # 针对不同提供商的配置调整
        if self.provider == 'nvidia':
            temperature = kwargs.get('temperature', 0.6)
            top_p = kwargs.get('top_p', 0.7)
            data.update({
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 4096
            })
            
        reasoning_content = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API请求失败: HTTP {response.status}\n{error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    async for line in response.content.iter_lines():
                        line_str = line.decode('utf-8')
                        if not line_str.strip() or not line_str.startswith('data:'):
                            continue
                        
                        data_json = line_str[5:].strip()
                        if data_json == "[DONE]":
                            continue
                            
                        try:
                            data = json.loads(data_json)
                            if not data.get("choices"):
                                continue
                                
                            delta = data["choices"][0].get("delta", {})
                            
                            # 根据推理模式提取内容
                            if self.reasoning_mode == 'reasoning_field':
                                reasoning = data["choices"][0].get("reasoning_content")
                                if reasoning:
                                    reasoning_content.append(reasoning)
                            elif self.reasoning_mode == 'think_tags':
                                content = delta.get("content", "")
                                if "<think>" in content:
                                    reasoning = await self.extract_reasoning_from_think_tags(content)
                                    if reasoning:
                                        reasoning_content.append(reasoning)
                            else:  # auto or any_content
                                content = delta.get("content", "")
                                if content:
                                    reasoning_content.append(content)
                        except json.JSONDecodeError:
                            continue
                            
            return "".join(reasoning_content)
        except Exception as e:
            logger.error(f"获取推理内容失败: {e}", exc_info=True)
            raise
```

```python
# app/deepclaude/reasoning/factory.py
from .base import BaseReasoningProvider
from .deepseek import DeepSeekReasoningProvider
from .ollama import OllamaReasoningProvider
import os
from app.utils.logger import logger

class ReasoningProviderFactory:
    """推理提供者工厂"""
    
    @staticmethod
    def create(provider_type: str = None) -> BaseReasoningProvider:
        """创建推理提供者实例"""
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
            raise ValueError(f"不支持的推理提供者类型: {provider_type}")
```

#### 工具模块

```python
# app/deepclaude/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class ToolProcessor(ABC):
    """工具处理器基类"""
    
    @abstractmethod
    def validate_and_convert(self, tools: List[Dict], target_format: str) -> List[Dict]:
        """验证并转换工具格式"""
        pass
```

```python
# app/deepclaude/tools/validators.py
from typing import Dict, List, Any, Optional
from app.utils.logger import logger

class ToolValidator:
    """工具验证器，用于验证工具格式的有效性"""
    
    @staticmethod
    def is_valid_openai_function(tool: Dict) -> bool:
        """验证是否为有效的OpenAI函数工具格式"""
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
        """验证是否为有效的Claude自定义工具格式"""
        if not isinstance(tool, dict):
            return False
            
        if "type" not in tool or tool["type"] != "custom":
            return False
            
        if "name" not in tool:
            return False
            
        return True
        
    @staticmethod
    def has_nested_custom_type(tool: Dict) -> bool:
        """检查Claude自定义工具中是否有嵌套的type字段"""
        if not isinstance(tool, dict) or "type" not in tool or tool["type"] != "custom":
            return False
            
        custom_field = tool.get("custom", {})
        return isinstance(custom_field, dict) and "type" in custom_field
```

```python
# app/deepclaude/tools/converters.py
from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator

class ToolConverter:
    """工具转换器，用于在不同格式间转换工具定义"""
    
    @staticmethod
    def openai_to_claude(tool: Dict) -> Dict:
        """将OpenAI格式工具转换为Claude格式"""
        if not ToolValidator.is_valid_openai_function(tool):
            logger.warning(f"工具格式错误，无法转换: {tool}")
            return None
            
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
        
        return claude_tool
        
    @staticmethod
    def claude_to_openai(tool: Dict) -> Dict:
        """将Claude格式工具转换为OpenAI格式"""
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
        """修复Claude自定义工具中的嵌套type字段问题"""
        if not ToolValidator.is_valid_claude_custom_tool(tool):
            return tool
            
        if ToolValidator.has_nested_custom_type(tool):
            fixed_tool = tool.copy()
            fixed_tool["custom"] = tool["custom"].copy()
            fixed_tool["custom"].pop("type", None)
            logger.debug(f"已修复工具中的嵌套type字段: {tool.get('name', '未命名工具')}")
            return fixed_tool
            
        return tool
```

```python
# app/deepclaude/tools/handlers.py
from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator
from .converters import ToolConverter
import uuid

class ToolHandler:
    """工具处理器，用于处理工具调用和结果"""
    
    @staticmethod
    def validate_and_convert_tools(tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
        """验证并转换工具格式"""
        if not tools:
            return []
            
        valid_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                logger.warning(f"工具格式错误: {tool}")
                continue
                
            # 处理已经是Claude格式的工具
            if "type" in tool and tool["type"] in ["custom", "bash_20250124", "text_editor_20250124"]:
                if tool["type"] == "custom":
                    # 修复可能存在的嵌套type字段问题
                    fixed_tool = ToolConverter.fix_claude_custom_tool(tool)
                    valid_tools.append(fixed_tool)
                else:
                    valid_tools.append(tool)
                logger.info(f"检测到已是Claude格式的工具: {tool.get('name', '未命名工具')}")
                continue
                
            # 处理OpenAI格式工具
            if "function" in tool:
                if target_format == 'claude-3':
                    claude_tool = ToolConverter.openai_to_claude(tool)
                    if claude_tool:
                        valid_tools.append(claude_tool)
                        logger.info(f"将OpenAI格式工具转换为Claude格式: {claude_tool.get('name', '未命名工具')}")
                else:
                    if "type" not in tool:
                        tool = {"type": "function", "function": tool["function"]}
                    valid_tools.append(tool)
                    logger.info(f"保持OpenAI格式工具: {tool['function'].get('name', '未命名工具')}")
                continue
                
            # 处理简化格式工具
            if "name" in tool and "parameters" in tool:
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
                
            logger.warning(f"工具格式无法识别: {tool}")
            
        logger.info(f"工具验证和转换完成，原有 {len(tools)} 个工具，有效 {len(valid_tools)} 个工具")
        return valid_tools
    
    @staticmethod
    def format_tool_call_for_streaming(tool_call: Dict, chat_id: str = None, created_time: int = None) -> Dict:
        """格式化工具调用为流式响应格式"""
        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        function_args = function.get("arguments", "{}")
        
        response = {
            "id": chat_id or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": created_time or 0,
            "model": "deepclaude",
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
        
        return response
```

### 4. 核心协调类

```python
# app/deepclaude/core.py
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import uuid
import time
from app.utils.logger import logger
from .reasoning.factory import ReasoningProviderFactory
from .tools.handlers import ToolHandler
from app.clients.claude_client import ClaudeClient

class DeepClaude:
    """DeepClaude核心协调类，整合推理与生成功能"""
    
    def __init__(self, **kwargs):
        logger.info("初始化DeepClaude服务...")
        
        # 配置参数
        self.claude_api_key = kwargs.get('claude_api_key', os.getenv('CLAUDE_API_KEY', ''))
        self.claude_api_url = kwargs.get('claude_api_url', os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com/v1/messages'))
        self.claude_provider = kwargs.get('claude_provider', os.getenv('CLAUDE_PROVIDER', 'anthropic'))
        self.is_origin_reasoning = kwargs.get('is_origin_reasoning', os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true')
        self.min_reasoning_chars = 100
        
        # 初始化组件
        self.claude_client = ClaudeClient(
            api_key=self.claude_api_key,
            api_url=self.claude_api_url,
            provider=self.claude_provider
        )
        
        self.tool_handler = ToolHandler()
        
        # 初始化推理提供者
        provider_type = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
        self.thinker_client = ReasoningProviderFactory.create(provider_type)
        
        # 数据库存储配置
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
        """处理流式聊天请求"""
        chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
        created_time = kwargs.get("created_time", int(time.time()))
        model = kwargs.get("model", "deepclaude")
        claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
        deepseek_model = kwargs.get("deepseek_model", "deepseek-reasoner")
        
        try:
            logger.info("开始流式处理请求...")
            
            # 如果有工具且直接透传模式开启，则直接使用Claude处理
            direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
            if direct_tool_pass and tools and len(tools) > 0:
                logger.info(f"直接透传模式(流式): 包含 {len(tools)} 个工具")
                
                # 验证并转换工具格式
                converted_tools = self.tool_handler.validate_and_convert_tools(tools, target_format='claude-3')
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
                
                # 准备Claude API调用参数
                claude_kwargs = {
                    "messages": messages,
                    "model": claude_model,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "tools": converted_tools
                }
                
                # 配置工具选择策略
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        claude_kwargs["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        logger.info("检测到'none'工具选择策略，将不使用工具")
                        claude_kwargs.pop("tools")
                elif isinstance(tool_choice, dict):
                    claude_kwargs["tool_choice"] = tool_choice
                
                # 直接流式调用Claude并透传响应
                async for content_type, content in self.claude_client.stream_chat(**claude_kwargs):
                    if content_type == "content":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            # "choices": [{
                            #     "index": 0,
                            #     "delta": {
                            #         "role": "assistant",
                            #         "content": content
                            #     }
                            # }]
```python
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
                    elif content_type == "tool_call":
                        tool_call_response = self.tool_handler.format_tool_call_for_streaming(
                            content, chat_id=chat_id, created_time=created_time
                        )
                        yield f"data: {json.dumps(tool_call_response, ensure_ascii=False)}\n\n".encode("utf-8")
                        
                yield f"data: [DONE]\n\n".encode("utf-8")
                return
            
            # 推理-生成模式处理
            original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
            
            # 保存对话到数据库
            if self.save_to_db:
                try:
                    user_id = None
                    title = original_question[:20] + "..." if original_question else None
                    self.current_conversation_id = self.db_ops.create_conversation(
                        user_id=user_id,
                        title=title
                    )
                    logger.info(f"创建新对话，ID: {self.current_conversation_id}")
                    self.db_ops.add_conversation_history(
                        conversation_id=self.current_conversation_id,
                        role="user",
                        content=original_question
                    )
                except Exception as db_e:
                    logger.error(f"保存对话数据失败: {db_e}")
            
            # 获取推理内容
            logger.info("正在获取推理内容...")
            try:
                reasoning = await self.thinker_client.get_reasoning(
                    messages=messages,
                    model=deepseek_model
                )
            except Exception as e:
                logger.error(f"获取推理内容失败: {e}")
                reasoning = "无法获取推理内容"
            
            # 输出推理过程
            if reasoning:
                reasoning_response = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "is_reasoning": True,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"🤔 思考过程:\n{reasoning}\n",
                            "reasoning": True
                        }
                    }]
                }
                yield f"data: {json.dumps(reasoning_response, ensure_ascii=False)}\n\n".encode("utf-8")
            
            # 处理工具调用
            has_tool_decision = False
            if tools and len(tools) > 0:
                try:
                    decision_prompt = self._format_tool_decision_prompt(original_question, reasoning, tools)
                    logger.debug(f"工具决策提示: {decision_prompt[:200]}...")
                    
                    tool_decision_response = await self.claude_client.chat(
                        messages=[{"role": "user", "content": decision_prompt}],
                        model=claude_model,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.9)
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
            
            # 如果没有使用工具，正常生成文本回答
            if not has_tool_decision:
                combined_prompt = f"我已经思考了以下问题：\n\n{original_question}\n\n我的思考过程是：\n{reasoning}\n\n现在，给出清晰、准确、有帮助的回答，不要提及上面的思考过程，直接开始回答。"
                claude_messages = [{"role": "user", "content": combined_prompt}]
                
                logger.info("正在获取Claude回答...")
                
                full_content = ""
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model=claude_model,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9)
                ):
                    if content_type in ["answer", "content"]:
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }]
                        }
                        full_content += content
                        yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode("utf-8")
                
                # 保存回答到数据库
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
                
                yield f"data: [DONE]\n\n".encode("utf-8")
        
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
        """处理非流式聊天请求"""
        logger.info("开始处理非流式请求...")
        
        try:
            # 直接工具透传模式
            direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
            if direct_tool_pass and tools and len(tools) > 0:
                logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
                
                # 验证并转换工具
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
                
                # 准备Claude API调用参数
                claude_kwargs = {
                    "messages": messages,
                    "model": claude_model,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "tools": converted_tools,
                    "stream": False
                }
                
                # 配置工具选择策略
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        claude_kwargs["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        logger.info("检测到'none'工具选择策略，将不使用工具")
                        claude_kwargs.pop("tools")
                elif isinstance(tool_choice, dict):
                    claude_kwargs["tool_choice"] = tool_choice
                
                # 调用Claude API
                response = await self.claude_client.chat(**claude_kwargs)
                return response
            
            # 推理-回答模式
            original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
            
            # 获取推理内容
            logger.info("正在获取推理内容...")
            reasoning = await self.thinker_client.get_reasoning(
                messages=messages,
                model=deepseek_model,
                model_arg=model_arg
            )
            logger.debug(f"获取到推理内容: {reasoning[:200] if reasoning else '无'}...")
            
            # 生成最终回答
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
            
            # 保存到数据库
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
        """格式化工具决策提示"""
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
        
        prompt = f"""用户问题：{original_question}

我的思考过程：
{reasoning}

可用工具：
{tools_description}

1. 仔细分析用户问题和思考过程。
2. 判断是否需要使用工具来回答问题。
3. 如果需要使用工具，请使用最合适的工具并提供所有必要的参数。
4. 如果不需要使用工具，直接回答用户问题。"""

        return prompt
```

### 5. 工具函数与辅助模块

```python
# app/deepclaude/utils/prompts.py
"""提供各种提示词模板"""

class PromptTemplates:
    """提示词模板集合"""
    
    @staticmethod
    def reasoning_prompt(question: str) -> str:
        """生成推理提示模板"""
        return f"""请思考下面这个问题，给出详细的分析过程：

{question}

分析思路：
"""

    @staticmethod
    def tool_decision_prompt(question: str, reasoning: str, tools_description: str) -> str:
        """生成工具决策提示模板"""
        return f"""用户问题：{question}

我的思考过程：
{reasoning}

可用工具：
{tools_description}

1. 仔细分析用户问题和思考过程。
2. 判断是否需要使用工具来回答问题。
3. 如果需要使用工具，请使用最合适的工具并提供所有必要的参数。
4. 如果不需要使用工具，直接回答用户问题。"""

    @staticmethod
    def final_answer_prompt(question: str, reasoning: str, tool_results: str = None) -> str:
        """生成最终回答提示模板"""
        tool_part = f"\n\n工具调用结果：\n{tool_results}" if tool_results else ""
        
        return f"""用户问题：{question}

我的思考过程：
{reasoning}{tool_part}

请根据以上信息，给出清晰、准确、有帮助的回答。不要在回答中提及你的思考过程或工具调用细节，直接回答用户问题。"""
```

```python
# app/deepclaude/utils/streaming.py
"""流式响应处理工具"""

import json
from typing import Dict, Any

class StreamingHelper:
    """流式响应辅助工具"""
    
    @staticmethod
    def format_chunk_response(content: str, role: str = "assistant", chat_id: str = None, 
                             created_time: int = None, model: str = "deepclaude", 
                             is_reasoning: bool = False, finish_reason: str = None) -> str:
        """格式化流式响应块"""
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
        """生成流式响应结束标记"""
        return "data: [DONE]\n\n"
```

## 四、重构关键错误修复点汇总

1. **工具验证与转换**：
   - 修复`_validate_and_convert_tools`中的`KeyError: 'custom'`错误
   - 实现严格的类型检查和安全的字段访问
   - 将工具验证逻辑拆分为单独的模块，提高可维护性

2. **流式处理优化**：
   - 使用明确的接口定义和错误处理
   - 规范流式响应格式，确保与OpenAI和Claude API兼容
   - 简化流程代码，减少嵌套层级

3. **推理提供者抽象**：
   - 使用策略模式隔离不同推理服务的实现细节
   - 通过工厂方法根据配置动态选择实现
   - 明确定义接口协议，便于添加新的推理提供者

4. **错误恢复机制**：
   - 添加合理的回退策略确保服务可用性
   - 加强日志记录，便于排查问题
   - 统一错误处理流程，提供用户友好的错误信息

## 五、重构实施步骤

1. 创建新的项目结构和目录
2. 实现各个接口定义和基础类
3. 逐步实现各个功能模块：
   - 先实现工具验证和转换模块
   - 然后实现推理提供者
   - 最后实现核心协调类
4. 编写单元测试确保各组件正常工作
5. 逐步替换原有实现，确保功能平滑过渡

## 六、重构建议

1. **当即可应用的修复**：先修复`_validate_and_convert_tools`方法中的`KeyError: 'custom'`错误：

```python
# 将这行代码
if tool["type"] == "custom" and isinstance(tool.get("custom", {}), dict) and "type" in tool["custom"]:

# 修改为
custom_field = tool.get("custom", {})
if tool["type"] == "custom" and isinstance(custom_field, dict) and "type" in custom_field:
```

2. **逐步拆分模块**：按照重构方案逐步将功能拆分到独立模块，每次修改后确保系统仍能正常工作。

3. **增强测试覆盖**：为每个重构的模块添加单元测试，确保重构不引入新问题。

4. **文档更新**：随着重构进展，更新项目文档，确保团队成员理解新的架构和设计模式。

通过这次重构，DeepClaude项目将获得更好的模块化结构，更清晰的责任分离，以及更强的可维护性和扩展性。同时，关键的Bug将得到修复，提高系统的稳定性和可靠性。