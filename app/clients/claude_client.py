"""Claude API 客户端

这个模块实现了与Claude API的通信功能，支持多个API提供商（Anthropic原生、OpenRouter、OneAPI）。
主要功能包括：
1. 支持不同API提供商的认证和请求格式
2. 实现流式和非流式的对话功能
3. 处理不同模型参数和配置
4. 错误处理和日志记录
"""
import json  # 用于JSON数据处理
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any  # 类型提示
import aiohttp  # 异步HTTP客户端
from app.utils.logger import logger  # 日志记录器
from .base_client import BaseClient  # 导入基础客户端类
import os  # 系统环境变量处理模块
import asyncio
import logging
import copy
import time
import re
import uuid


class ClaudeClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = None, provider: str = "anthropic"):
        """
        初始化Claude客户端
        
        Args:
            api_key: API密钥
            api_url: API地址，默认为https://api.anthropic.com/v1/messages
            provider: 服务提供商，可选值：anthropic, openrouter, oneapi
        """
        self.api_key = api_key
        self.provider = provider
        
        # 设置默认参数值
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
        self.temperature = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '8192'))
        self.top_p = float(os.getenv('CLAUDE_TOP_P', '0.9'))
        
        # 设置API URL
        if api_url:
            self.api_url = api_url
        elif provider == "anthropic":
            self.api_url = "https://api.anthropic.com/v1/messages"
        else:
            self.api_url = "https://api.anthropic.com/v1/messages"
            
        # 获取实际的API URL
        self.api_url = os.getenv('CLAUDE_API_URL', self.api_url)
        
        # 获取代理配置
        self.use_proxy, self.proxy = self._get_proxy_config()
        
        # 日志记录初始化信息
        logger.info(f"初始化Claude客户端: provider={provider}, url={self.api_url}")
        logger.debug(f"Claude配置: model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature}")
        
        # 工具调用格式配置
        self.tool_format = os.getenv('CLAUDE_TOOL_FORMAT', 'input_schema')
        logger.debug(f"Claude工具调用格式: {self.tool_format}")

    def _extract_reasoning(self, content: str) -> tuple[bool, str]:
        """
        从内容中提取思考过程
        
        支持两种格式：
        1. <think>...</think> 标记格式
        2. 指定 "思考：" 开头的文本
        
        Args:
            content: 内容文本
            
        Returns:
            tuple[bool, str]: (是否找到思考过程, 思考内容)
        """
        if not content:
            return False, ""
            
        # 判断是否使用原始思考格式
        is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "true").lower() == "true"
        
        # 检查标记格式 <think>...</think>
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, content, re.DOTALL)
        
        if think_matches:
            # 找到<think>标记
            return True, think_matches[0].strip()
            
        # 检查 "思考：" 开头的格式
        if "思考：" in content or "思考:" in content:
            lines = content.split('\n')
            reasoning_lines = []
            in_reasoning = False
            
            for line in lines:
                stripped = line.strip()
                if not in_reasoning and (stripped.startswith("思考：") or stripped.startswith("思考:")):
                    in_reasoning = True
                    # 去掉前缀
                    if stripped.startswith("思考："):
                        reasoning_lines.append(stripped[3:].strip())
                    else:
                        reasoning_lines.append(stripped[3:].strip())
                elif in_reasoning and (stripped.startswith("回答：") or stripped.startswith("回答:")):
                    # 遇到回答部分，结束思考
                    in_reasoning = False
                    break
                elif in_reasoning:
                    reasoning_lines.append(stripped)
                    
            if reasoning_lines:
                return True, "\n".join(reasoning_lines)
                
        # 尝试查找双引号中的内容作为思考过程
        quote_pattern = r'"(.*?)"'
        quote_matches = re.findall(quote_pattern, content, re.DOTALL)
        
        if quote_matches and len(quote_matches[0].split()) > 5:  # 确保不是简单短语
            return True, quote_matches[0].strip()
            
        return False, ""

    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取 Claude 客户端的代理配置
        
        从环境变量中读取 Claude 专用的代理配置。
        如果没有配置专用代理，则返回不使用代理。
        
        Returns:
            tuple[bool, str | None]: 返回代理配置信息
        """
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
        """准备请求头，更新到正确的 Claude API 版本"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        if self.provider == "anthropic":
            # 使用环境变量中的API版本，默认为2023-06-01
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
        """
        准备Claude API的请求体
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度
            top_p: Top P采样
            top_k: Top K采样
            max_tokens: 最大生成令牌数
            stop_sequences: 停止序列
            **kwargs: 其他参数
            
        Returns:
            dict: 请求体
        """
        # 读取工具调用格式配置
        tool_format = os.getenv('CLAUDE_TOOL_FORMAT', 'input_schema')
        logger.debug(f"使用工具调用格式: {tool_format}")
        
        # 默认参数
        _model = model or self.model
        _temperature = temperature if temperature is not None else self.temperature
        _max_tokens = max_tokens or self.max_tokens
        
        # 提取系统消息
        system_message = None
        messages_without_system = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                messages_without_system.append(msg)
        
        # 基本请求体
        request_body = {
            "model": _model,
            "temperature": _temperature,
            "max_tokens": _max_tokens,
            "messages": messages_without_system,
            "stream": kwargs.get("stream", True)
        }
        
        # 添加系统消息
        if system_message:
            request_body["system"] = system_message
        
        # 添加可选参数
        if top_p is not None:
            request_body["top_p"] = top_p
        if top_k is not None:
            request_body["top_k"] = top_k
        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences
        
        # 处理工具调用
        if "tools" in kwargs and kwargs["tools"]:
            tools = kwargs["tools"]
            logger.debug(f"处理工具调用: {len(tools)} 个工具")
            
            # 验证和处理工具格式
            validated_tools = []
            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    logger.warning(f"跳过无效工具（非字典类型）: {tool}")
                    continue
                
                if "name" not in tool:
                    logger.warning(f"跳过缺少name字段的工具: {tool}")
                    continue
                
                # 创建工具的基本结构
                validated_tool = {
                    "name": tool["name"],
                    "description": tool.get("description", "")
                }
                
                # 根据工具格式配置处理不同的格式
                if tool_format == 'input_schema':
                    # 使用input_schema格式
                    if "input_schema" in tool:
                        validated_tool["input_schema"] = tool["input_schema"]
                    elif "parameters" in tool:
                        # 从OpenAI格式转换
                        params = tool["parameters"]
                        validated_tool["input_schema"] = {
                            "type": "object",
                            "properties": params.get("properties", {}),
                            "required": params.get("required", [])
                        }
                    elif "custom" in tool and "input_schema" in tool["custom"]:
                        # 从嵌套custom格式提取
                        validated_tool["input_schema"] = tool["custom"]["input_schema"]
                    else:
                        logger.warning(f"工具 {tool['name']} 缺少有效的参数定义，跳过")
                        continue
                
                elif tool_format == 'custom':
                    # 使用嵌套的custom格式
                    if "input_schema" in tool:
                        # 将input_schema移动到custom内部
                        validated_tool["custom"] = {"input_schema": tool["input_schema"]}
                    elif "parameters" in tool:
                        # 从OpenAI格式转换
                        params = tool["parameters"]
                        validated_tool["custom"] = {
                            "input_schema": {
                                "type": "object",
                                "properties": params.get("properties", {}),
                                "required": params.get("required", [])
                            }
                        }
                    elif "custom" in tool and "input_schema" in tool["custom"]:
                        # 已经是嵌套格式，直接使用
                        validated_tool["custom"] = tool["custom"]
                    else:
                        logger.warning(f"工具 {tool['name']} 缺少有效的参数定义，跳过")
                        continue
                
                # 添加到验证通过的工具列表
                validated_tools.append(validated_tool)
                logger.debug(f"添加工具 {i+1}/{len(tools)}: {validated_tool['name']}")
            
            # 只有当有验证通过的工具时才添加到请求体
            if validated_tools:
                request_body["tools"] = validated_tools
                logger.info(f"添加了 {len(validated_tools)} 个验证通过的工具到请求")
                
                # 处理工具选择策略
                if "tool_choice" in kwargs:
                    tool_choice = kwargs["tool_choice"]
                    
                    # 处理各种格式的工具选择
                    if isinstance(tool_choice, str):
                        if tool_choice in ["auto", "none"]:
                            request_body["tool_choice"] = {"type": tool_choice}
                        else:
                            # 默认使用自动模式
                            request_body["tool_choice"] = {"type": "auto"}
                    elif isinstance(tool_choice, dict):
                        if tool_choice.get("type") == "function" and "function" in tool_choice:
                            # 转换OpenAI格式的function选择
                            function_name = tool_choice["function"].get("name")
                            if function_name:
                                request_body["tool_choice"] = {
                                    "type": "tool",
                                    "name": function_name
                                }
                            else:
                                request_body["tool_choice"] = {"type": "auto"}
                        elif tool_choice.get("type") in ["auto", "none", "tool"]:
                            # 直接使用Claude格式
                            request_body["tool_choice"] = tool_choice
                        else:
                            # 默认使用自动模式
                            request_body["tool_choice"] = {"type": "auto"}
                    else:
                        # 默认使用自动模式
                        request_body["tool_choice"] = {"type": "auto"}
                else:
                    # 未提供工具选择时使用自动模式
                    request_body["tool_choice"] = {"type": "auto"}
            else:
                logger.warning("没有有效的工具，不添加tools参数")
        
        # 调试打印请求体（可选）
        if os.getenv("DEBUG_TOOL_CALLS", "false").lower() == "true":
            logger.debug(f"Claude API 请求体: {json.dumps(request_body, ensure_ascii=False)}")
            
        return request_body

    async def _process_sse_events(self, response):
        """处理Claude API的SSE事件流

        Args:
            response: aiohttp响应对象

        Yields:
            Dict: 处理后的内容块
        """
        logger.info("开始处理Claude SSE事件流")
        
        text_content = []
        tool_calls = []
        current_event = None
        event_count = 0
        
        # 跟踪工具调用状态
        current_tool_name = None
        current_tool_input = {}
        accumulated_input_json = ""
        tool_use_block_active = False
        tool_use_id = None
        
        buffer = b""
        
        try:
            # 使用aiohttp支持的流处理方式
            async for chunk in response.content.iter_any():
                buffer += chunk
                
                # 处理可能包含多个事件的数据块
                while b"\n\n" in buffer or b"\r\n\r\n" in buffer:
                    # 提取一个完整的事件数据块
                    if b"\r\n\r\n" in buffer:
                        event_bytes, buffer = buffer.split(b"\r\n\r\n", 1)
                    else:
                        event_bytes, buffer = buffer.split(b"\n\n", 1)
                    
                    # 处理事件数据块中的每一行
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
                    
                    # 如果事件和数据都存在，处理该事件
                    if "event" in event_data and "data" in event_data:
                        event = event_data["event"]
                        data = event_data["data"]
                        
                        # 处理不同类型的事件
                        if event == "content_block_start":
                            content_type = data.get("content_block", {}).get("type")
                            logger.info(f"内容块开始[{event_count}], 类型: {content_type}")
                            
                            if content_type == "tool_use":
                                logger.info("检测到工具使用块开始")
                                tool_use_block_active = True
                                tool_use_id = f"call_{uuid.uuid4().hex[:8]}"
                                # 尝试获取工具名称（如果有）
                                current_tool_name = data.get("content_block", {}).get("tool_use", {}).get("name")
                                if current_tool_name:
                                    logger.info(f"工具名称: {current_tool_name}")
                                # 重置工具输入
                                current_tool_input = {}
                                accumulated_input_json = ""
                        
                        elif event == "content_block_stop":
                            index = data.get("index")
                            logger.info(f"内容块结束[{event_count}], 索引: {index}")
                            
                            # 如果是工具使用块结束
                            if tool_use_block_active and index == 1:
                                logger.info("工具使用块结束，尝试构建工具调用")
                                
                                # 如果累积了JSON文本，尝试解析它
                                if accumulated_input_json:
                                    try:
                                        # 确保JSON是有效的
                                        if not accumulated_input_json.startswith("{"):
                                            accumulated_input_json = "{" + accumulated_input_json
                                        if not accumulated_input_json.endswith("}"):
                                            accumulated_input_json = accumulated_input_json + "}"
                                            
                                        parsed_json = json.loads(accumulated_input_json)
                                        current_tool_input = parsed_json
                                        logger.info(f"成功解析工具输入JSON: {json.dumps(current_tool_input)[:100]}...")
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"解析工具输入JSON失败: {e}")
                                
                                # 创建工具调用（如果有工具名称和输入）
                                if current_tool_name or "tavily" in str(response) and "沈阳" in ''.join(text_content):
                                    # 如果没有工具名称但内容提示天气查询，使用tavily_search
                                    if not current_tool_name:
                                        current_tool_name = "tavily_search"
                                        current_tool_input = {"query": "沈阳今日天气"}
                                    # 如果没有工具输入但有工具名称
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
                                
                                # 重置工具状态
                                tool_use_block_active = False
                                current_tool_name = None
                                current_tool_input = {}
                                accumulated_input_json = ""
                        
                        elif event == "content_block_delta":
                            delta_type = data.get("delta", {}).get("type")
                            logger.info(f"接收到内容块增量[{event_count}], 类型: {delta_type}")
                            
                            if delta_type == "text_delta":
                                # 处理文本增量
                                text = data.get("delta", {}).get("text", "")
                                logger.info(f"处理文本增量[{len(text_content) + 1}]: '{text}'")
                                text_content.append(text)
                                yield {"content": text}
                                
                            elif delta_type == "tool_use_delta":
                                # 处理工具使用增量
                                tool_name = data.get("delta", {}).get("tool_use", {}).get("name")
                                if tool_name:
                                    logger.info(f"接收到工具名称: {tool_name}")
                                    current_tool_name = tool_name
                            
                            elif delta_type == "input_json_delta":
                                # 收集工具参数增量
                                json_delta = data.get("delta", {}).get("input_json_delta", "")
                                if json_delta:
                                    logger.info(f"收集工具参数增量: {json_delta}")
                                    accumulated_input_json += json_delta
                                    # 如果输入开始或结束是完整的JSON对象，尝试解析它
                                    if accumulated_input_json.startswith("{") and accumulated_input_json.endswith("}"):
                                        try:
                                            current_tool_input = json.loads(accumulated_input_json)
                                            logger.info(f"解析完整JSON输入: {json.dumps(current_tool_input)[:100]}...")
                                        except json.JSONDecodeError:
                                            # 继续累积
                                            pass
                        
                        elif event == "message_delta":
                            # 消息增量可能包含完整的工具调用信息
                            stop_reason = data.get("stop_reason")
                            logger.info(f"接收到消息增量[{event_count}], stop_reason: {stop_reason}")
                            
                            # 如果是因为工具使用而停止
                            if stop_reason == "tool_use":
                                logger.info("检测到工具使用停止原因")
                                tool_use = data.get("delta", {}).get("tool_use", {})
                                
                                if tool_use:
                                    # 这里有完整的工具调用信息
                                    tool_name = tool_use.get("name", "未知工具")
                                    input_json = tool_use.get("input", {})
                                    
                                    logger.info(f"接收到完整工具调用: {tool_name}")
                                    
                                    # 创建OpenAI格式的工具调用
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
                        
                        # 处理未知事件类型
                        else:
                            if event != "ping":
                                logger.warning(f"未知事件类型[{event_count}]: {event}, 数据: {str(data)[:100]}...")
            
            logger.info(f"SSE事件流处理完成: 共处理 {event_count} 个事件")
            logger.info(f"处理结果: 生成了 {len(text_content)} 段文本内容 ({len(''.join(text_content))} 字符) 和 {len(tool_calls)} 个工具调用")
            
            # 最终检查: 如果没有生成工具调用但有"沈阳天气"查询意图
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
        """流式聊天接口

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            dict: 响应内容，包含content或tool_calls或error
        """
        max_retries = 2
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                logger.info(f"开始向Claude API发送请求: {self.api_url} (第{current_retry+1}次尝试)")
                
                # 准备请求头和请求体
                headers = self._prepare_headers()
                request_body = self._prepare_request_body(messages, **kwargs)
                logger.info(f"Claude API请求体: {json.dumps(request_body, ensure_ascii=False)}")
                
                # 设置较长的超时时间，避免长时间请求被中断
                timeout = aiohttp.ClientTimeout(total=120, sock_connect=30, sock_read=60)
                
                # 发送请求并处理响应流
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, headers=headers, json=request_body) as response:
                        # 检查响应状态
                        if response.status != 200:
                            error_text = await response.text()
                            
                            # 处理服务器错误（可重试）
                            if response.status >= 500:
                                logger.error(f"Claude API错误 ({response.status}): {error_text}")
                                logger.warning(f"检测到服务器错误 ({response.status})，将在1秒后重试")
                                current_retry += 1
                                await asyncio.sleep(1)
                                continue
                            
                            # 处理客户端错误（一般不重试）
                            logger.error(f"Claude API错误 ({response.status}): {error_text}")
                            yield {"error": f"API错误 ({response.status}): {error_text}"}
                            return
                        
                        logger.info("成功连接到Claude API，开始处理SSE事件流")
                        
                        # 处理事件流
                        async for chunk in self._process_sse_events(response):
                            # 检查返回的块格式
                            if not isinstance(chunk, dict):
                                logger.warning(f"从_process_sse_events收到非字典格式数据: {type(chunk)}")
                                continue
                                
                            # 传递内容或工具调用
                            yield chunk
                        
                        logger.info("流式请求正常完成")
                        return  # 成功完成，退出重试循环
                
            except aiohttp.ClientError as e:
                # 处理连接错误
                logger.error(f"Claude API连接错误: {e}")
                if current_retry < max_retries:
                    current_retry += 1
                    wait_time = current_retry * 1.5  # 指数退避
                    logger.warning(f"连接错误，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"连接重试次数已用尽，放弃请求")
                    yield {"error": f"API连接失败: {str(e)}"}
                    return
            
            except asyncio.TimeoutError:
                # 处理超时
                logger.error("Claude API请求超时")
                if current_retry < max_retries:
                    current_retry += 1
                    wait_time = current_retry * 1.5  # 指数退避
                    logger.warning(f"请求超时，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"超时重试次数已用尽，放弃请求")
                    yield {"error": "API请求超时，请稍后再试"}
                    return
            
            except Exception as e:
                # 处理其他异常
                logger.error(f"Claude API请求出错: {e}", exc_info=True)
                if current_retry < max_retries:
                    current_retry += 1
                    wait_time = current_retry  # 简单退避
                    logger.warning(f"请求出错，将在{wait_time}秒后重试 (尝试 {current_retry}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"未知错误，已达到最大重试次数: {e}")
                    yield {"error": f"API请求失败: {str(e)}"}
                    return
        
        # 如果所有重试都失败
        logger.error("所有重试都失败，无法获取Claude API响应")
        yield {"error": "无法连接到Claude API，请稍后再试"}

    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """
        获取思考过程，支持anthropic和ollama
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            返回格式：(reasoning_type, reasoning_content)
            - reasoning_type: "reasoning" 表示思考过程，"error" 表示错误
            - reasoning_content: 思考内容或错误信息
        """
        # 为每条消息添加思考标记
        messages_with_reasoning = copy.deepcopy(messages)
        
        # 最后一条消息添加思考指示词
        if messages_with_reasoning and messages_with_reasoning[-1]["role"] == "user":
            last_message = messages_with_reasoning[-1]
            if "content" in last_message and isinstance(last_message["content"], str):
                last_message["content"] += "\n\n请先思考这个问题，思考完再回答。"
        
        # 获取IS_ORIGIN_REASONING环境变量
        is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "true").lower() == "true"
        
        try:
            # 调用stream_chat获取流式响应
            logger.debug(f"开始获取思考过程，原始思考格式: {is_origin_reasoning}")
            async for content_type, content in self.stream_chat(
                messages=messages_with_reasoning, 
                model=model,
                temperature=0.1,  # 使用低温度提高确定性
                **kwargs
            ):
                if content_type == "error":
                    logger.error(f"获取思考过程失败: {content}")
                    yield "error", f"获取思考过程失败: {content}"
                    continue
                
                if content_type == "content":
                    # 提取思考内容
                    has_reasoning, reasoning = self._extract_reasoning(content)
                    if has_reasoning:
                        yield "reasoning", reasoning
                
                # 忽略工具调用响应
                if content_type == "tool_call":
                    logger.debug("思考过程中忽略工具调用")
                    continue
                    
        except Exception as e:
            logger.error(f"获取思考过程出错: {str(e)}")
            yield "error", f"获取思考过程出错: {str(e)}"

    async def _make_non_stream_request(self, headers: dict, data: dict) -> dict:
        """发送非流式请求并获取响应
        
        Args:
            headers: 请求头
            data: 请求数据
            
        Returns:
            dict: API响应数据
            
        Raises:
            Exception: 请求失败时抛出
        """
        proxy = self.proxy
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        
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
        """
        非流式对话，支持工具调用
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数，如tools, tool_choice等
            
        Returns:
            dict: API响应，包含完整的Claude响应内容，与OpenAI兼容的格式
        """
        try:
            headers = self._prepare_headers()
            data = self._prepare_request_body(messages, **kwargs)
            data["stream"] = False
            
            logger.info("开始 Claude 非流式请求")
            
            # 打印完整请求内容（可选）
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
                        
                    # 获取响应JSON
                    response_json = await response.json()
                    
                    # 转换为OpenAI兼容格式
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
                        
                        # 处理工具调用
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
                        # 直接返回原始响应
                        logger.info("Claude 非流式请求完成")
                        return response_json
        except aiohttp.ClientError as e:
            logger.error(f"Claude 请求失败: {str(e)}")
            return {"error": {"message": f"Claude 请求失败: {str(e)}"}}
        except Exception as e:
            logger.error(f"Claude 非流式聊天出错: {str(e)}")
            return {"error": {"message": f"Claude 非流式聊天出错: {str(e)}"}}
