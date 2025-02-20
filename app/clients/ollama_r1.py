"""Ollama R1 API 客户端

这个模块实现了与本地Ollama服务的通信功能，支持推理过程和最终结果的流式输出。
主要功能包括：
1. 支持流式对话
2. 处理特殊的思考标签（think tags）
3. 错误处理和日志记录
"""
import os
import json
from typing import AsyncGenerator
from app.utils.logger import logger
from .base_client import BaseClient
import asyncio

class OllamaR1Client(BaseClient):
    def __init__(self, api_url: str = "http://localhost:11434"):
        """
        初始化 Ollama R1 客户端
        
        专注于本地运行的 DeepSeek R1 模型
        只支持 think 标签格式
        
        Args:
            api_url: Ollama服务地址，默认为本地地址
        """
        if not api_url:
            raise ValueError("必须提供 Ollama API URL")
            
        if not api_url.endswith("/api/chat"):
            api_url = f"{api_url.rstrip('/')}/api/chat"
            
        super().__init__(api_key="", api_url=api_url)
        self.default_model = "deepseek-r1:32b"
    
    def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
        """
        处理包含 think 标签的内容
        
        分析和处理文本中的思考标签（<think>和</think>），用于区分模型的思考过程
        和最终输出。这个方法会检查标签的完整性，确保正确处理部分接收到的内容。
        
        Args:
            content: 需要处理的内容字符串，可能包含think标签
            
        Returns:
            tuple[bool, str]: 
                bool: 是否检测到完整的think标签对（同时包含开始和结束标签）
                str: 处理后的内容文本
        """
        has_start = "<think>" in content
        has_end = "</think>" in content
        
        if has_start and has_end:
            return True, content
        elif has_start:
            return False, content
        elif not has_start and not has_end:
            return False, content
        else:
            return True, content
            
    def _extract_reasoning(self, content: str) -> tuple[bool, str]:
        """从 think 标签提取思考内容"""
        if "<think>" in content and "</think>" in content:
            start = content.find("<think>") + 7
            end = content.find("</think>")
            if start < end:
                return True, content[start:end].strip()
        return False, ""

    async def stream_chat(self, messages: list, model: str = "deepseek-r1:32b") -> AsyncGenerator[tuple[str, str], None]:
        # 添加参数验证
        if not messages:
            raise ValueError("消息列表不能为空")
            
        headers = {
            "Content-Type": "application/json",
        }
        
        # Ollama API格式转换
        data = {
            "model": model,
            "messages": messages,  # 使用完整的消息历史
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            }
        }
        
        logger.debug(f"开始流式对话：{data}")
        
        try:
            current_content = ""
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                try:
                    response = json.loads(chunk_str)
                    if "message" in response and "content" in response["message"]:
                        content = response["message"]["content"]
                        current_content += content
                        
                        # 使用 _extract_reasoning 提取推理内容
                        has_reasoning, reasoning = self._extract_reasoning(current_content)
                        if has_reasoning:
                            yield "reasoning", reasoning
                            # 清理已处理的内容
                            current_content = current_content[current_content.find("</think>") + 8:]
                            
                    if response.get("done"):
                        # 最后检查一次
                        has_reasoning, reasoning = self._extract_reasoning(current_content)
                        if has_reasoning:
                            yield "reasoning", reasoning
                        return
                        
                except json.JSONDecodeError:
                    continue
                
        except Exception as e:
            logger.error(f"流式对话发生错误: {e}", exc_info=True)
            raise
    async def get_reasoning(self, messages: list, model: str = None, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取思考过程
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称，默认使用 default_model
            **kwargs: 额外参数
            
        Yields:
            tuple[str, str]: (content_type, content)
                - content_type: "reasoning" 表示思考过程
                - content: 具体内容
        """
        if model is None:
            model = self.default_model
            
        async for content_type, content in self.stream_chat(
            messages=messages,
            model=model
        ):
            if content_type == "reasoning":
                yield content_type, content
    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取代理配置"""
        proxy = os.getenv("OLLAMA_PROXY")
        if proxy:
            logger.info(f"Ollama 客户端使用代理: {proxy}")
        else:
            logger.debug("Ollama 客户端未启用代理")
        return bool(proxy), proxy