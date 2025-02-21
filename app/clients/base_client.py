"""基础客户端类，定义通用接口

这个模块定义了一个抽象基类，用于实现与不同AI模型API的通信。
它提供了基础的HTTP请求处理和流式响应处理功能，子类可以通过继承该类来实现具体的API调用逻辑。
"""
from typing import AsyncGenerator, Any, Tuple
import aiohttp  # 异步HTTP客户端库
from app.utils.logger import logger  # 日志记录器
from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()


class BaseClient(ABC):
    def __init__(self, api_key: str, api_url: str):
        """初始化基础客户端
        
        初始化客户端所需的基本配置，包括API密钥和API地址。
        这些配置将用于后续的API请求。
        
        Args:
            api_key: API密钥，用于API认证
            api_url: API服务器地址，用于发送请求的目标URL
        """
        self.api_key = api_key
        self.api_url = api_url
        
    def _get_proxy(self) -> str | None:
        """获取代理配置"""
        use_proxy, proxy = self._get_proxy_config()
        return proxy if use_proxy else None

    @abstractmethod
    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取代理配置
        Returns:
            tuple[bool, str | None]: (是否使用代理, 代理地址)
        """
        pass
        
    @abstractmethod
    async def stream_chat(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话接口"""
        pass
        
    @abstractmethod
    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取推理过程
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称
            **kwargs: 额外参数，包括：
                - is_origin_reasoning (bool): 是否使用原始推理
                - model_arg (tuple): 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)
        """
        pass
    
    @abstractmethod
    def _extract_reasoning(self, content: str) -> tuple[bool, str]:
        """从内容中提取推理过程
        
        Args:
            content: 原始内容
            
        Returns:
            tuple[bool, str]: (是否包含完整推理, 推理内容)
        """
        pass
    
    async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
        max_retries = 3
        retry_count = 0
        retry_codes = {429, 500, 502, 503, 504}  # 可重试的状态码
        
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
                                wait_time = min(2 ** retry_count, 32)  # 最大等待32秒
                                logger.warning(f"等待 {wait_time} 秒后重试...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                            raise Exception(f"HTTP {response.status}: {error_msg}")
                            
                        # 使用更高效的缓冲区处理
                        buffer = bytearray()
                        async for chunk in response.content.iter_chunks():
                            chunk_data = chunk[0]  # iter_chunks 返回 (data, end_of_http_chunk) 元组
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
                await asyncio.sleep(wait_time)
