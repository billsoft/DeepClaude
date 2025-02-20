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
import os

class DeepSeekClient(BaseClient):
    def __init__(self, api_key: str, api_url: str = None, provider: str = None):
        """初始化 DeepSeek 客户端
        
        支持多个提供商:
        - deepseek: 官方API
        - siliconflow: 硅基云API
        - nvidia: NVIDIA API
        """
        self.provider = provider or os.getenv('DEEPSEEK_PROVIDER', 'deepseek')
        
        # 各提供商的默认配置
        self.provider_configs = {
            'deepseek': {
                'url': 'https://api.deepseek.com/v1/chat/completions',
                'model': 'deepseek-reasoner'
            },
            'siliconflow': {
                'url': 'https://api.siliconflow.cn/v1/chat/completions',
                'model': 'deepseek-ai/DeepSeek-R1'
            },
            'nvidia': {
                'url': 'https://integrate.api.nvidia.com/v1/chat/completions',
                'model': 'deepseek-ai/deepseek-r1'
            }
        }
        
        if self.provider not in self.provider_configs:
            raise ValueError(f"不支持的 provider: {self.provider}")
            
        config = self.provider_configs[self.provider]
        api_url = api_url or os.getenv('DEEPSEEK_API_URL') or config['url']
        super().__init__(api_key, api_url)
        
        self.default_model = config['model']
        self.is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true'
        
    def _get_proxy_config(self) -> tuple[bool, str | None]:
        """获取 DeepSeek 客户端的代理配置
        
        从环境变量中读取 DeepSeek 专用的代理配置。
        如果没有配置专用代理，则返回不使用代理。
        
        Returns:
            tuple[bool, str | None]: 返回代理配置信息
        """
        enable_proxy = os.getenv('DEEPSEEK_ENABLE_PROXY', 'false').lower() == 'true'
        if enable_proxy:
            http_proxy = os.getenv('HTTP_PROXY')
            https_proxy = os.getenv('HTTPS_PROXY')
            logger.info(f"DeepSeek 客户端使用代理: {https_proxy or http_proxy}")
            return True, https_proxy or http_proxy
        logger.debug("DeepSeek 客户端未启用代理")
        return False, None
    
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
            
    def _extract_reasoning(self, content: str) -> tuple[bool, str]:
        """提取推理内容
        
        支持两种格式:
        1. reasoning_content字段 (is_origin_reasoning=true)
        2. think标签 (is_origin_reasoning=false)
        """
        if self.is_origin_reasoning:
            if "reasoning_content" in content:
                return True, content["reasoning_content"]
            return False, ""
        else:
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>") + 7
                end = content.find("</think>")
                if start < end:
                    return True, content[start:end].strip()
            return False, ""

    async def stream_chat(self, messages: list, model: str = None, model_arg: tuple = None) -> AsyncGenerator[tuple[str, str], None]:
        """基础的流式对话方法"""
        if not model:
            model = self.default_model
        
        if not model:
            raise ValueError("未指定模型且无默认模型")
        
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
        
        # NVIDIA 特定参数
        if self.provider == 'nvidia':
            temperature = model_arg[0] if model_arg else 0.6
            top_p = model_arg[1] if model_arg else 0.7
            data.update({
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 4096
            })
            
        logger.debug(f"开始流式对话：{data}")
        
        try:
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                try:
                    data = json.loads(chunk_str)
                    if not data or not data.get("choices") or not data["choices"][0].get("delta"):
                        continue
                        
                    delta = data["choices"][0]["delta"]
                    
                    # 使用 _extract_reasoning 提取推理内容
                    has_reasoning, reasoning = self._extract_reasoning(delta)
                    if has_reasoning:
                        yield "reasoning", reasoning
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.error(f"流式对话发生错误: {e}", exc_info=True)
            raise
            
    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取推理过程
        
        根据配置使用不同的推理提取方式:
        1. 原始推理格式: 通过 reasoning_content 字段获取
        2. 标签格式: 通过 <think></think> 标签获取
        """
        model_arg = kwargs.get('model_arg')
        
        # 构建请求头和数据
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
        
        # NVIDIA 特定参数
        if self.provider == 'nvidia':
            temperature = model_arg[0] if model_arg else 0.6
            top_p = model_arg[1] if model_arg else 0.7
            data.update({
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 4096
            })
        
        reasoning_buffer = []
        content_buffer = []
        
        async for chunk in self._make_request(headers, data):
            try:
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                for line in chunk_str.splitlines():
                    if line.startswith("data: "):
                        json_str = line[len("data: "):]
                        if json_str == "[DONE]":
                            continue
                        
                        data = json.loads(json_str)
                        if not data or not data.get("choices") or not data["choices"][0].get("delta"):
                            continue
                        
                        delta = data["choices"][0]["delta"]
                        content = delta.get("content", "")
                        
                        if self.is_origin_reasoning:
                            # 原始推理格式
                            if "reasoning_content" in delta:
                                yield "reasoning", delta["reasoning_content"]
                        else:
                            # 标签格式
                            if "<think>" in content:
                                # 提取 think 标签中的内容
                                start = content.find("<think>") + len("<think>")
                                end = content.find("</think>")
                                if end > start:
                                    reasoning = content[start:end].strip()
                                    if reasoning:
                                        yield "reasoning", reasoning
                            elif content:
                                yield "content", content
                            
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"处理推理内容时发生错误: {e}")
                continue
