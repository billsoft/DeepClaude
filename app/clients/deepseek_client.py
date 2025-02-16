"""DeepSeek API 客户端

这个模块实现了与DeepSeek API的通信功能，支持推理过程和最终结果的流式输出。
主要功能包括：
1. 支持流式对话和推理过程的实时展示
2. 处理特殊的思考标签（think tags）
3. 支持多种输出模式（原始推理/普通对话）
4. 错误处理和日志记录
"""
import json  # 用于JSON数据处理
from typing import AsyncGenerator  # 异步生成器类型
from app.utils.logger import logger  # 日志记录器
from .base_client import BaseClient  # 导入基础客户端类

VALID_MODELS = ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-Chat-7B"]

class DeepSeekClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = "https://api.siliconflow.cn/v1/chat/completions", provider: str = "deepseek"):
        """初始化 DeepSeek 客户端
        
        配置与DeepSeek API通信所需的基本参数，包括API密钥和服务器地址。
        默认使用DeepSeek官方API地址。
        
        Args:
            api_key: DeepSeek API密钥，用于API认证
            api_url: DeepSeek API地址，默认使用官方API地址
            provider: API提供商，默认为"deepseek"
        """
        super().__init__(api_key, api_url)
        self.provider = provider
        self.default_model = "deepseek-ai/DeepSeek-R1"  # 添加默认模型
        
    def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
        """处理包含 think 标签的内容
        
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
            
    async def stream_chat(self, messages: list, model: str = "deepseek-ai/DeepSeek-R1", is_origin_reasoning: bool = True) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话
        
        实现与DeepSeek API的流式对话功能，支持以下特性：
        1. 实时输出模型的推理过程和最终回答
        2. 支持原始推理模式和普通对话模式
        3. 自动处理流式响应和特殊标签
        4. 错误处理和日志记录
        
        Args:
            messages: 消息列表，包含对话历史和当前输入
            model: 模型名称，默认使用DeepSeek-R1模型
            is_origin_reasoning: 是否使用原始推理模式，默认为True
                - True: 分别输出推理过程和最终答案
                - False: 只输出普通对话内容
            
        Yields:
            tuple[str, str]: 返回(内容类型, 内容)的元组
                内容类型: 
                    - "reasoning" - 表示模型的推理过程
                    - "content" - 表示模型的最终答案
                内容: 实际的文本内容
        """
        if model not in VALID_MODELS:
            error_msg = f"无效的模型名称: {model}，可用模型: {VALID_MODELS}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        
        logger.debug(f"开始流式对话：{data}")
        
        try:
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                for line in chunk_str.splitlines():
                    if line.startswith("data: "):
                        json_str = line[len("data: "):]
                        if json_str == "[DONE]":
                            return
                        
                        try:
                            data = json.loads(json_str)
                            if not data or not data.get("choices") or not data["choices"][0].get("delta"):
                                continue
                            
                            delta = data["choices"][0]["delta"]
                            if is_origin_reasoning:
                                # 处理原始推理内容
                                if delta.get("reasoning_content"):
                                    content = delta["reasoning_content"]
                                    logger.debug(f"提取推理内容：{content}")
                                    yield "reasoning", content
                                elif delta.get("content"):
                                    content = delta["content"]
                                    logger.info(f"提取内容信息，推理阶段结束: {content}")
                                    yield "content", content
                            else:
                                # 处理普通对话内容
                                if delta.get("content"):
                                    content = delta["content"]
                                    yield "content", content
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON 解析错误: {e}")
                            continue
        except Exception as e:
            logger.error(f"流式对话发生错误: {e}", exc_info=True)
            raise
