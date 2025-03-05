from .base import BaseGenerationProvider
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import aiohttp
from app.utils.logger import logger
from app.clients.claude_client import ClaudeClient

class ClaudeGenerationProvider(BaseGenerationProvider):
    """基于Claude的生成提供者"""
    
    def __init__(self, api_key: str, api_url: str = None, provider: str = "anthropic"):
        """初始化Claude生成提供者
        
        Args:
            api_key: Claude API密钥
            api_url: Claude API地址，如果为None则使用默认地址
            provider: 提供商类型，支持anthropic/openrouter/oneapi
        """
        super().__init__(api_key, api_url)
        self.provider = provider.lower()
        self.client = ClaudeClient(
            api_key=api_key,
            api_url=api_url,
            provider=provider
        )
        logger.info(f"初始化Claude生成提供者: provider={self.provider}")
    
    async def generate_response(self, messages: List[Dict], model: str = None, **kwargs) -> Dict:
        """生成回答内容
        
        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认值
            **kwargs: 其他参数
            
        Returns:
            回答内容
        """
        try:
            # 如果未指定模型，使用环境变量或默认值
            if not model:
                model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
                
            response = await self.client.chat(
                messages=messages,
                model=model,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                stream=False,
                tools=kwargs.get('tools'),
                tool_choice=kwargs.get('tool_choice')
            )
            
            return response
        except Exception as e:
            logger.error(f"生成回答内容失败: {e}", exc_info=True)
            raise
    
    async def stream_response(self, messages: List[Dict], model: str = None, **kwargs) -> AsyncGenerator[Tuple[str, str], None]:
        """流式生成回答内容
        
        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认值
            **kwargs: 其他参数
            
        Yields:
            (内容类型, 内容)元组，内容类型可以是"content"或"tool_call"
        """
        try:
            # 如果未指定模型，使用环境变量或默认值
            if not model:
                model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
                
            async for content_type, content in self.client.stream_chat(
                messages=messages,
                model=model,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                tools=kwargs.get('tools'),
                tool_choice=kwargs.get('tool_choice')
            ):
                yield content_type, content
        except Exception as e:
            logger.error(f"流式生成回答内容失败: {e}", exc_info=True)
            raise