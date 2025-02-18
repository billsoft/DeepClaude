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
        self.provider = provider
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
            return True, https_proxy or http_proxy
        return False, None

    async def stream_chat(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        model: str,
        stream: bool = True
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式或非流式对话
        
        实现与Claude API的对话功能，支持以下特性：
        1. 支持流式和非流式输出
        2. 自动适配不同API提供商的请求格式
        3. 支持自定义模型参数
        4. 错误处理和日志记录
        
        Args:
            messages: 消息列表，包含对话历史和当前输入
            model_arg: 模型参数元组[temperature, top_p, presence_penalty, frequency_penalty]
                - temperature: 温度参数，控制输出的随机性，范围[0,1]
                - top_p: 核采样参数，控制输出的多样性
                - presence_penalty: 存在惩罚，降低重复token的概率
                - frequency_penalty: 频率惩罚，降低高频token的概率
            model: 模型名称，对于OpenRouter会自动转换为对应格式
            stream: 是否使用流式输出，默认为True
            
        Yields:
            tuple[str, str]: 返回(内容类型, 内容)的元组
                内容类型: "answer" - 表示模型的回答
                内容: 实际的文本内容
        
        Raises:
            ValueError: 当提供的API提供商不支持时抛出
        """

        if self.provider == "openrouter":
            # 转换模型名称为 OpenRouter 格式
            model = "anthropic/claude-3.5-sonnet"
                
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ErlichLiu/DeepClaude",  # OpenRouter 需要
                "X-Title": "DeepClaude"  # OpenRouter 需要
            }

            data = {
                "model": model,  # OpenRouter 使用 anthropic/claude-3.5-sonnet 格式
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
                "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0], # Claude仅支持temperature与top_p
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
                        json_str = line[6:]  # 去掉 'data: ' 前缀
                        if json_str.strip() == '[DONE]':
                            return
                            
                        try:
                            data = json.loads(json_str)
                            if self.provider in ("openrouter", "oneapi"):
                                # OpenRouter/OneApi 格式
                                content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    yield "answer", content
                            elif self.provider == "anthropic":
                                # Anthropic 格式
                                if data.get('type') == 'content_block_delta':
                                    content = data.get('delta', {}).get('text', '')
                                    if content:
                                        yield "answer", content
                            else:
                                raise ValueError(f"不支持的Claude Provider: {self.provider}")
                        except json.JSONDecodeError:
                            continue
        else:
            # 非流式输出
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
                    continue
