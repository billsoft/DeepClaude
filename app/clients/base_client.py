"""基础客户端类，定义通用接口

这个模块定义了一个抽象基类，用于实现与不同AI模型API的通信。
它提供了基础的HTTP请求处理和流式响应处理功能，子类可以通过继承该类来实现具体的API调用逻辑。
"""
from typing import AsyncGenerator, Any  # 类型提示，用于异步生成器和任意类型
import aiohttp  # 异步HTTP客户端库
from app.utils.logger import logger  # 日志记录器
from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器


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
        
    async def _make_request(self, headers: dict, data: dict) -> AsyncGenerator[bytes, None]:
        """发送 API 请求并处理响应
        
        Args:
            headers: 请求头
            data: 请求数据
            
        Yields:
            bytes: 响应数据流
        """
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
        """流式对话，由子类实现
        
        这是一个抽象方法，需要由具体的子类实现。用于处理与特定AI模型的流式对话。
        子类实现时需要处理：
        1. 消息格式转换
        2. API特定参数配置
        3. 响应解析和处理
        4. 错误处理
        
        Args:
            messages: 消息列表，包含对话历史和当前输入
            model: 模型名称，指定使用的AI模型
            
        Yields:
            tuple[str, str]: 返回(内容类型, 内容)的元组，用于区分不同类型的响应内容
        """
        pass
