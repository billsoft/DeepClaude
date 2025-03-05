from .base import BaseGenerationProvider
from .claude import ClaudeGenerationProvider
import os
from app.utils.logger import logger

class GenerationProviderFactory:
    """生成提供者工厂类"""
    
    @staticmethod
    def create(provider_type: str = None) -> BaseGenerationProvider:
        """创建生成提供者实例
        
        Args:
            provider_type: 提供者类型，目前仅支持claude
            
        Returns:
            生成提供者实例
        
        Raises:
            ValueError: 如果指定了不支持的提供者类型
        """
        provider_type = provider_type or os.getenv('GENERATION_PROVIDER', 'claude').lower()
        
        if provider_type == 'claude':
            api_key = os.getenv('CLAUDE_API_KEY')
            api_url = os.getenv('CLAUDE_API_URL')
            claude_provider = os.getenv('CLAUDE_PROVIDER', 'anthropic')
            if not api_key:
                raise ValueError("未设置CLAUDE_API_KEY环境变量")
            return ClaudeGenerationProvider(
                api_key=api_key, 
                api_url=api_url, 
                provider=claude_provider
            )
        else:
            raise ValueError(f"不支持的生成提供者类型: {provider_type}") 