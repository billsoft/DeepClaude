from ..interfaces import GenerationProvider
from abc import abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional

class BaseGenerationProvider(GenerationProvider):
    """生成提供者基类"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """初始化生成提供者
        
        Args:
            api_key: API密钥，可选
            api_url: API地址，可选
        """
        self.api_key = api_key
        self.api_url = api_url
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """生成回答内容的抽象方法，必须由子类实现
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            回答内容
        """
        pass
    
    @abstractmethod
    async def stream_response(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
        """流式生成回答内容的抽象方法，必须由子类实现
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Yields:
            (内容类型, 内容)元组
        """
        pass 