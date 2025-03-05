from .base import BaseReasoningProvider
from typing import Dict, List, Any
import json
import aiohttp
from app.utils.logger import logger

class OllamaReasoningProvider(BaseReasoningProvider):
    """基于Ollama的推理提供者"""
    
    def __init__(self, api_url: str = "http://localhost:11434/api/chat"):
        super().__init__(api_key=None, api_url=api_url)
        self.model = "deepseek-chat"
        
    async def get_reasoning(self, messages: List[Dict], model: str, **kwargs) -> str:
        """获取Ollama推理内容"""
        data = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API请求失败: HTTP {response.status}\n{error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    if "message" in result:
                        return result["message"].get("content", "")
                    else:
                        logger.warning(f"Ollama响应缺少消息内容: {result}")
                        return ""
        except Exception as e:
            logger.error(f"获取Ollama推理内容失败: {e}", exc_info=True)
            raise 