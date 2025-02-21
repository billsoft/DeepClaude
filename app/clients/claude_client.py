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
        """准备请求头
        
        根据不同的 provider 准备对应的请求头
        """
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
        """准备请求数据
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        """
        data = {
            "model": kwargs.get("model", "claude-3-5-sonnet-20241022"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 8192),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": True
        }
        
        # 记录请求数据
        logger.debug(f"Claude请求数据: {messages}")
        
        return data

    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[dict, None]:
        """流式对话"""
        try:
            headers = self._prepare_headers()
            data = self._prepare_request_data(messages, **kwargs)
            
            logger.debug(f"Claude请求数据: {data}")
            
            async for chunk in self._make_request(headers, data):
                try:
                    if chunk:
                        # 解析 SSE 数据
                        text = chunk.decode('utf-8')
                        if text.startswith('data: '):
                            data = text[6:].strip()
                            if data == '[DONE]':
                                break
                                
                            response = json.loads(data)
                            logger.debug(f"Claude响应数据: {response}")
                            
                            # 检查响应结构
                            if 'type' in response:  # Claude API 格式
                                if response['type'] == 'content_block_delta':
                                    content = response['delta'].get('text', '')
                                    if content:
                                        yield {"delta": {"content": content}}
                            elif 'choices' in response:  # OpenAI 格式
                                if response['choices'][0].get('delta', {}).get('content'):
                                    yield response['choices'][0]
                                    
                except json.JSONDecodeError as e:
                    logger.error(f"解析Claude响应失败: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Claude流式请求失败: {e}", exc_info=True)
            raise

    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """Claude 不提供推理过程
        
        这个方法仅用于满足基类接口要求。Claude 只负责根据思考过程生成最终答案。
        
        Yields:
            tuple[str, str]: 空生成器，因为 Claude 不提供推理
        """
        # Claude 不提供推理过程
        return
        yield  # 为了满足生成器语法要求
