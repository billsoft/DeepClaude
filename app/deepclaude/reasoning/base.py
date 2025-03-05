from ..interfaces import ReasoningProvider
from abc import abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional

class BaseReasoningProvider(ReasoningProvider):
    """推理提供者基类"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """初始化推理提供者
        
        Args:
            api_key: API密钥，可选
            api_url: API地址，可选
        """
        self.api_key = api_key
        self.api_url = api_url
        
    async def extract_reasoning_content(self, raw_content: str) -> str:
        """提取推理内容的方法，可被子类重写
        
        Args:
            raw_content: 原始内容
            
        Returns:
            提取后的推理内容
        """
        return raw_content
        
    @abstractmethod
    async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
        """获取推理内容的抽象方法，必须由子类实现
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            推理内容
        """
        pass 