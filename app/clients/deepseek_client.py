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

        accumulated_content = ""
        is_collecting_think = False
        
        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode('utf-8')
            
            try:
                lines = chunk_str.splitlines()
                for line in lines:
                    if line.startswith("data: "):
                        json_str = line[len("data: "):]
                        if json_str == "[DONE]":
                            return
                        
                        data = json.loads(json_str)
                        if data and data.get("choices") and data["choices"][0].get("delta"):
                            delta = data["choices"][0]["delta"]
                            
                            if is_origin_reasoning:
                                # 处理 reasoning_content
                                if delta.get("reasoning_content"):
                                    content = delta["reasoning_content"]
                                    logger.debug(f"提取推理内容：{content}")
                                    yield "reasoning", content
                                
                                if delta.get("reasoning_content") is None and delta.get("content"):
                                    content = delta["content"]
                                    logger.info(f"提取内容信息，推理阶段结束: {content}")
                                    yield "content", content
                            else:
                                # 处理其他模型的输出
                                if delta.get("content"):
                                    content = delta["content"]
                                    if content == "":  # 只跳过完全空的字符串
                                        continue
                                    logger.debug(f"非原生推理内容：{content}")
                                    accumulated_content += content
                                    
                                    # 检查累积的内容是否包含完整的 think 标签对
                                    is_complete, processed_content = self._process_think_tag_content(accumulated_content)
                                    
                                    if "<think>" in content and not is_collecting_think:
                                        # 开始收集推理内容
                                        logger.debug(f"开始收集推理内容：{content}")
                                        is_collecting_think = True
                                        yield "reasoning", content
                                    elif is_collecting_think:
                                        if "</think>" in content:
                                            # 推理内容结束
                                            logger.debug(f"推理内容结束：{content}")
                                            is_collecting_think = False
                                            yield "reasoning", content
                                            # 输出空的 content 来触发 Claude 处理
                                            yield "content", ""
                                            # 重置累积内容
                                            accumulated_content = ""
                                        else:
                                            # 继续收集推理内容
                                            yield "reasoning", content
                                    else:
                                        # 普通内容
                                        yield "content", content
                                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
            except Exception as e:
                logger.error(f"处理 chunk 时发生错误: {e}")
