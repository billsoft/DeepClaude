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
    
    async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
        try:
            # 获取代理配置
            use_proxy, proxy = self._get_proxy_config()
            
            # 创建 TCP 连接器，设置更长的超时时间
            connector = aiohttp.TCPConnector(
                ssl=False,
                force_close=True,
                enable_cleanup_closed=True  # 添加自动清理
            )
            
            # 设置更合理的超时时间
            timeout = aiohttp.ClientTimeout(
                total=120,  # 总超时时间
                connect=30,  # 连接超时
                sock_read=60  # 读取超时
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                logger.debug(f"正在发送请求到: {self.api_url}")
                logger.debug(f"使用代理: {proxy if use_proxy else '不使用代理'}")
                
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    proxy=proxy if use_proxy else None,
                    timeout=timeout  # 使用新的超时设置
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"API请求失败: HTTP {response.status}\n{error_text}"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    
                    async for chunk in response.content.iter_any():
                        if chunk:
                            yield chunk
                        
        except asyncio.TimeoutError as e:
            logger.error(f"请求超时: {e}")
            raise
        except Exception as e:
            logger.error(f"请求发生错误: {e}")
            raise
