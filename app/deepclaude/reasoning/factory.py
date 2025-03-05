from .base import BaseReasoningProvider
from .deepseek import DeepSeekReasoningProvider
from .ollama import OllamaReasoningProvider
import os
from app.utils.logger import logger

class ReasoningProviderFactory:
    """推理提供者工厂"""
    
    @staticmethod
    def create(provider_type: str = None) -> BaseReasoningProvider:
        """创建推理提供者实例"""
        provider_type = provider_type or os.getenv('REASONING_PROVIDER', 'deepseek').lower()
        
        if provider_type == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL')
            return DeepSeekReasoningProvider(api_key, api_url, provider='deepseek')
        elif provider_type == 'siliconflow':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.siliconflow.cn/v1/chat/completions')
            return DeepSeekReasoningProvider(api_key, api_url, provider='siliconflow')
        elif provider_type == 'nvidia':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL')
            return DeepSeekReasoningProvider(api_key, api_url, provider='nvidia')
        elif provider_type == 'ollama':
            api_url = os.getenv('OLLAMA_API_URL')
            return OllamaReasoningProvider(api_url=api_url)
        else:
            raise ValueError(f"不支持的推理提供者类型: {provider_type}") 