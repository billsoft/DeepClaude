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