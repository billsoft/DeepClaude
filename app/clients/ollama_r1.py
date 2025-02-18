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

class OllamaR1Client(BaseClient):
    def __init__(self, api_url: str = "http://localhost:11434"):
        """
        初始化 Ollama R1 客户端
        
        配置与本地Ollama服务通信所需的基本参数，包括服务器地址。
        默认使用本地Ollama服务地址。
        
        Args:
            api_url: Ollama服务地址，默认为本地地址
        """
        # 确保API URL以/api/chat结尾
        if not api_url.endswith("/api/chat"):
            api_url = f"{api_url.rstrip('/')}/api/chat"
            
        # Ollama本地服务不需要API密钥
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
            
    async def stream_chat(self, messages: list, model: str = "deepseek-r1:32b") -> AsyncGenerator[tuple[str, str], None]:
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
                    if response.get("done"):
                        # 最后一个chunk，确保处理完所有内容
                        if current_content:
                            is_complete, content = self._process_think_tag_content(current_content)
                            if is_complete:
                                # 提取<think>标签之间的内容
                                start_idx = content.find("<think>") + 7
                                end_idx = content.find("</think>")
                                yield "reasoning", content[start_idx:end_idx].strip()
                            else:
                                yield "content", content
                        return
                    
                    if "message" in response and "content" in response["message"]:
                        content = response["message"]["content"]
                        current_content += content
                        # 检查是否有完整的think标签对
                        is_complete, processed_content = self._process_think_tag_content(current_content)
                        if is_complete:
                            # 提取<think>标签之间的内容
                            start_idx = current_content.find("<think>") + 7
                            end_idx = current_content.find("</think>")
                            yield "reasoning", current_content[start_idx:end_idx].strip()
                            # 重置current_content为剩余的内容
                            current_content = current_content[end_idx + 8:].strip()
                        elif "<think>" not in current_content:
                            # 只有在确定没有任何think标签时才输出普通内容
                            yield "content", content
                            
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析错误: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"流式对话发生错误: {e}", exc_info=True)
            raise
    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取代理配置
        
        由于Ollama是本地服务，默认不需要代理。
        如果需要代理，可以通过OLLAMA_PROXY环境变量设置。
        
        Returns:
            tuple[bool, str | None]: 返回一个元组，包含：
                - bool: 是否启用代理
                - str | None: 代理地址，如果不启用代理则为None
        """
        proxy = os.getenv("OLLAMA_PROXY")
        return bool(proxy), proxy