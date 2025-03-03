"""Claude API 客户端

这个模块实现了与Claude API的通信功能，支持多个API提供商（Anthropic原生、OpenRouter、OneAPI）。
主要功能包括：
1. 支持不同API提供商的认证和请求格式
2. 实现流式和非流式的对话功能
3. 处理不同模型参数和配置
4. 错误处理和日志记录
"""
import json  # 用于JSON数据处理
from typing import AsyncGenerator  # 异步生成器类型
from app.utils.logger import logger  # 日志记录器
from .base_client import BaseClient  # 导入基础客户端类
import os  # 系统环境变量处理模块


class ClaudeClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages", provider: str = "anthropic"):
        """初始化 Claude 客户端
        
        根据不同的API提供商初始化客户端配置。支持三种提供商：
        - anthropic: Anthropic官方API
        - openrouter: OpenRouter代理API
        - oneapi: OneAPI代理服务
        
        Args:
            api_key: Claude API密钥，根据provider不同而不同
            api_url: Claude API地址，默认使用Anthropic官方API地址
            provider: API提供商，默认为"anthropic"
        """
        super().__init__(api_key, api_url)
        self.provider = provider.lower()

    def _extract_reasoning(self, content: str) -> tuple[bool, str]:
        """Claude 不需要提取推理过程，因为它是回答者
        
        Args:
            content: 原始内容
            
        Returns:
            tuple[bool, str]: 始终返回 (False, "") 因为 Claude 不负责推理
        """
        # Claude 是回答者而不是思考者，所以不需要提取推理过程
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
            headers.update({
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",  # 修复为正确的 API 版本
                "anthropic-beta": "messages-2023-12-15"  # 修复为正确的 beta 版本
            })
        elif self.provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "oneapi":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        logger.debug(f"Claude API 请求头: {headers}")
        return headers

    def _prepare_request_data(self, messages: list, **kwargs) -> dict:
        """准备请求数据，支持工具调用，添加详细日志"""
        data = {
            "model": kwargs.get("model", os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 8192),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": kwargs.get("stream", True)
        }
        
        # 添加工具相关参数，增加详细日志
        if "tools" in kwargs and kwargs["tools"]:
            tools = kwargs["tools"]
            valid_tools = []
            for tool in tools:
                if not isinstance(tool, dict) or "function" not in tool:
                    logger.warning(f"跳过无效的工具定义: {tool}")
                    continue
                valid_tools.append(tool)
                logger.info(f"工具验证通过: {tool['function'].get('name', '未命名工具')}")
                logger.debug(f"工具详情: {json.dumps(tool, ensure_ascii=False)}")
            
            if valid_tools:
                data["tools"] = valid_tools
                logger.info(f"向 Claude API 添加 {len(valid_tools)} 个有效工具")
            else:
                logger.warning("没有有效的工具可以添加到请求中")
        
        if "tool_choice" in kwargs:
            tool_choice = kwargs["tool_choice"]
            if tool_choice not in ["auto", "none"]:
                logger.warning(f"不支持的工具选择策略: {tool_choice}，使用默认值 'auto'")
                tool_choice = "auto"
            data["tool_choice"] = tool_choice
            logger.info(f"工具选择策略设置为: {tool_choice}")
        
        logger.debug(f"Claude API 请求数据: {json.dumps(data, ensure_ascii=False)}")
        return data

    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[dict, None]:
        """流式对话，支持工具调用，增加详细日志"""
        try:
            headers = self._prepare_headers()
            data = self._prepare_request_data(messages, **kwargs)
            
            logger.info("开始 Claude 流式请求")
            logger.info(f"请求包含工具: {'是' if 'tools' in data else '否'}")
            
            async for chunk in self._make_request(headers, data):
                try:
                    if chunk:
                        text = chunk.decode('utf-8')
                        if text.startswith('data: '):
                            data = text[6:].strip()
                            if data == '[DONE]':
                                logger.info("Claude 流式响应完成")
                                break
                                
                            response = json.loads(data)
                            logger.debug(f"Claude 响应数据: {json.dumps(response, ensure_ascii=False)}")
                            
                            # 处理不同类型的响应，增加详细日志
                            if 'type' in response:  # Claude API 格式
                                if response['type'] == 'content_block_delta':
                                    content = response['delta'].get('text', '')
                                    if content:
                                        logger.debug(f"收到内容块: {content[:50]}...")
                                        yield "content", content
                                elif response['type'] == 'tool_calls':
                                    # 处理工具调用，添加详细日志
                                    tool_calls = response.get('tool_calls', [])
                                    logger.info(f"Claude 决定使用 {len(tool_calls)} 个工具")
                                    for tool_call in tool_calls:
                                        logger.info(f"工具调用: {tool_call.get('function', {}).get('name', '未知工具')}")
                                        logger.debug(f"工具调用详情: {json.dumps(tool_call, ensure_ascii=False)}")
                                        yield "tool_call", tool_call
                                elif response['type'] == 'error':
                                    error_msg = response.get('error', {}).get('message', '未知错误')
                                    logger.error(f"Claude API 错误: {error_msg}")
                                    yield "error", error_msg
                            elif 'choices' in response:  # OpenAI 格式
                                choice = response['choices'][0]
                                if 'delta' in choice:
                                    delta = choice['delta']
                                    if 'content' in delta:
                                        logger.debug(f"收到内容: {delta['content'][:50]}...")
                                        yield "content", delta['content']
                                    elif 'tool_calls' in delta:
                                        logger.info("收到工具调用(OpenAI格式)")
                                        yield "tool_call", delta['tool_calls']
                            
                except json.JSONDecodeError as e:
                    logger.error(f"解析 Claude 响应失败: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Claude 流式请求失败: {e}", exc_info=True)
            raise

    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """Claude 不提供推理过程
        
        这个方法仅用于满足基类接口要求。Claude 只负责根据思考过程生成最终答案。
        
        Yields:
            tuple[str, str]: 空生成器，因为 Claude 不提供推理
        """
        # Claude 不提供推理过程，但为了与接口一致，我们可以返回一个空内容
        if kwargs.get('stream', True) == False:
            # 如果是非流式模式，返回一个"answer"类型的内容
            async for content_type, content in self.stream_chat(
                messages=messages,
                model=kwargs.get('model', os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')),
                **kwargs
            ):
                # 将所有内容收集起来
                all_content = content
                # 作为answer类型返回
                yield "answer", all_content
                return
        # 其他情况下，不产生任何内容        
        return

    async def chat(self, messages: list, **kwargs) -> dict:
        """非流式对话，支持工具调用"""
        try:
            headers = self._prepare_headers()
            data = self._prepare_request_data(messages, **kwargs)
            data["stream"] = False
            
            logger.info("开始Claude非流式请求")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            
            response = await self._make_request(headers, data)
            if isinstance(response, bytes):
                response = response.decode('utf-8')
                if response.startswith('data: '):
                    response = response[6:].strip()
                response = json.loads(response)
            
            logger.debug(f"Claude非流式响应: {json.dumps(response, ensure_ascii=False)}")
            
            # 检查工具调用响应
            if 'tool_calls' in response:
                logger.info(f"收到工具调用响应: {len(response['tool_calls'])} 个工具")
            
            return response
            
        except Exception as e:
            logger.error(f"Claude非流式请求失败: {e}", exc_info=True)
            raise
