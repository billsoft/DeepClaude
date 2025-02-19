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
        
        配置与本地Ollama服务通信所需的基本参数，包括服务器地址。
        默认使用本地Ollama服务地址。
        
        Args:
            api_url: Ollama服务地址，默认为本地地址
        """
        # 添加参数验证
        if not api_url:
            raise ValueError("必须提供 Ollama API URL")
            
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
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
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
                                    # 修复：确保有think标签时才提取推理内容
                                    if "<think>" in content and "</think>" in content:
                                        start_idx = content.find("<think>") + 7
                                        end_idx = content.find("</think>")
                                        if start_idx < end_idx:  # 确保标签位置正确
                                            yield "reasoning", content[start_idx:end_idx].strip()
                                    else:
                                        yield "content", content
                                else:
                                    yield "content", content
                            return
                        
                        if "message" in response and "content" in response["message"]:
                            content = response["message"]["content"]
                            current_content += content
                            
                            # 修复：改进think标签处理逻辑
                            if "<think>" in current_content and "</think>" in current_content:
                                # 找到所有完整的think标签对
                                while "<think>" in current_content and "</think>" in current_content:
                                    start_idx = current_content.find("<think>")
                                    end_idx = current_content.find("</think>")
                                    
                                    if start_idx < end_idx:
                                        # 提取并输出推理内容
                                        reasoning = current_content[start_idx + 7:end_idx].strip()
                                        if reasoning:
                                            yield "reasoning", reasoning
                                        
                                        # 更新current_content，移除已处理的部分
                                        current_content = current_content[end_idx + 8:].strip()
                                    else:
                                        break
                            
                            # 如果剩余内容中没有think标签，作为普通内容输出
                            if current_content and "<think>" not in current_content:
                                yield "content", current_content
                                current_content = ""
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 解析错误: {e}")
                        continue
                    
                break  # 成功则跳出循环
                
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count < max_retries:
                    delay = 2 ** retry_count  # 指数退避
                    logger.warning(f"请求超时，第 {retry_count} 次重试，等待 {delay} 秒...")
                    await asyncio.sleep(delay)
                    continue
                logger.error("达到最大重试次数，放弃请求")
                raise
            except Exception as e:
                logger.error(f"流式对话发生错误: {e}", exc_info=True)
                raise
    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取推理过程
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称
            **kwargs: 额外参数
            
        Yields:
            tuple[str, str]: (content_type, content)
                - content_type: "reasoning" 表示推理过程，"content" 表示最终答案
                - content: 具体内容
        """
        async for content_type, content in self.stream_chat(
            messages=messages,
            model=model
        ):
            if content_type in ("reasoning", "content"):
                yield content_type, content
    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取代理配置"""
        proxy = os.getenv("OLLAMA_PROXY")
        if proxy:
            logger.info(f"Ollama 客户端使用代理: {proxy}")
        else:
            logger.debug("Ollama 客户端未启用代理")
        return bool(proxy), proxy