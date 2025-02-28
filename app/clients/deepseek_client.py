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
import logging
import re

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
        
        # 设置推理内容提取模式
        # 可选值：
        # - 'auto': 自动检测API响应格式
        # - 'reasoning_field': 使用reasoning_content字段
        # - 'think_tags': 使用<think>标签
        # - 'any_content': 将所有内容视为推理内容
        # - 'early_content': 将第一部分内容作为推理，之后的作为答案
        self.reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto').lower()
        
        # 兼容旧环境变量
        self.is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true'
        if self.is_origin_reasoning and self.reasoning_mode == 'auto':
            self.reasoning_mode = 'reasoning_field'
        
        # 模式切换阈值
        self.early_content_threshold = int(os.getenv('DEEPSEEK_EARLY_THRESHOLD', '20'))
        
        # 缓存最近的内容以便更好地区分推理和回答
        self._content_buffer = ""
        self._reasoning_buffer = ""
        self._has_found_reasoning = False
        self._content_token_count = 0
            
        logger.debug(f"DeepSeek客户端初始化完成 - 提供商: {self.provider}, 模型: {self.default_model}, 推理模式: {self.reasoning_mode}")
        
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
            
    def _extract_reasoning(self, content: str | dict) -> tuple[bool, str]:
        """提取推理内容
        
        支持多种格式:
        1. reasoning_content字段 (reasoning_mode='reasoning_field')
        2. think标签 (reasoning_mode='think_tags')
        3. 任何内容 (reasoning_mode='any_content')
        4. 自动检测 (reasoning_mode='auto')
        5. 早期内容 (reasoning_mode='early_content')
        """
        # 详细记录输入内容类型和值
        logger.debug(f"提取推理内容，content类型: {type(content)}, 推理模式: {self.reasoning_mode}")
        
        # 处理字典类型
        if isinstance(content, dict):
            logger.debug(f"处理字典类型的推理内容: {str(content)[:100]}...")
            
            # 1. 处理reasoning_content字段 (官方原始推理模式)
            if "reasoning_content" in content:
                extracted = content["reasoning_content"]
                logger.debug(f"从reasoning_content字段提取到推理内容: {str(extracted)[:50]}...")
                return True, extracted
                
            # 2. 处理role字段，可能包含思考角色
            if "role" in content and content["role"] in ["reasoning", "thinking", "thought"]:
                if "content" in content:
                    logger.debug(f"从思考角色提取到推理内容")
                    return True, content["content"]
                
            # 3. 处理普通content字段
            if "content" in content:
                text_content = content["content"]
                
                # 3.1 如果内容包含think标签，尝试提取
                if self.reasoning_mode in ['auto', 'think_tags'] and "<think>" in text_content:
                    return self._extract_from_think_tags(text_content)
                    
                # 3.2 如果设置为任何内容都视为推理，直接返回
                if self.reasoning_mode in ['auto', 'any_content']:
                    logger.debug(f"任何内容模式，将普通内容视为推理: {text_content[:50]}...")
                    return True, text_content
                
                # 3.3 如果是早期内容模式且尚未收集足够的内容，视为推理
                if self.reasoning_mode == 'early_content' and self._content_token_count < self.early_content_threshold:
                    self._content_token_count += 1
                    logger.debug(f"早期内容模式，将内容视为推理 (token {self._content_token_count}/{self.early_content_threshold})")
                    return True, text_content
            
            # 4. 处理特殊模型输出 (如NVIDIA提供商可能有不同字段)
            if self.provider == 'nvidia' and self.reasoning_mode == 'auto':
                # 检查可能的特殊字段
                for field in ["thinking", "thought", "reasoning"]:
                    if field in content:
                        logger.debug(f"从NVIDIA特殊字段{field}提取到推理内容")
                        return True, content[field]
                        
            return False, ""
            
        # 处理字符串类型
        elif isinstance(content, str):
            logger.debug(f"处理字符串类型的推理内容: {content[:50]}...")
            
            # 1. 尝试从think标签提取
            if self.reasoning_mode in ['auto', 'think_tags']:
                # 更新累积缓冲区以处理跨多个块的标签
                self._content_buffer += content
                has_think, extracted = self._extract_from_buffered_think_tags()
                if has_think:
                    return True, extracted
                
            # 2. 如果是早期内容模式且尚未收集足够的内容，视为推理
            if self.reasoning_mode == 'early_content' and self._content_token_count < self.early_content_threshold:
                self._content_token_count += 1
                logger.debug(f"早期内容模式，将内容视为推理 (token {self._content_token_count}/{self.early_content_threshold})")
                return True, content
                
            # 3. 如果设置为任何内容都视为推理，直接返回
            if self.reasoning_mode in ['auto', 'any_content']:
                logger.debug(f"任何内容模式，将字符串内容视为推理: {content[:50]}...")
                return True, content
            
            # 4. 判断是否为可能的推理内容 (启发式识别)
            if self.reasoning_mode == 'auto' and self._is_potential_reasoning(content):
                logger.debug(f"根据启发式判断，将内容视为推理: {content[:50]}...")
                return True, content
                
            return False, ""
            
        logger.warning(f"无法处理的内容类型: {type(content)}")
        return False, ""
    
    def _is_potential_reasoning(self, text: str) -> bool:
        """使用启发式方法判断文本是否可能是推理内容"""
        # 如果已经找到过推理内容，增加连续性检查
        if self._has_found_reasoning:
            return True
            
        # 推理指示词模式
        reasoning_patterns = [
            r'我需要思考', r'让我分析', r'分析这个问题', r'思路：', r'思考过程',
            r'首先[，,]', r'第一步', r'第二步', r'第三步', r'接下来',
            r'算法思路', r'解题思路', r'考虑问题'
        ]
        
        # 检查模式
        for pattern in reasoning_patterns:
            if re.search(pattern, text):
                self._has_found_reasoning = True
                return True
                
        return False
        
    def _extract_from_buffered_think_tags(self) -> tuple[bool, str]:
        """从缓冲区中提取think标签内容"""
        buffer = self._content_buffer
        
        if "<think>" not in buffer:
            return False, ""
            
        # 完整标签对
        if "</think>" in buffer:
            start = buffer.find("<think>") + len("<think>")
            end = buffer.find("</think>")
            if start < end:
                extracted = buffer[start:end].strip()
                # 清空缓冲区
                self._content_buffer = buffer[end + len("</think>"):]
                logger.debug(f"从缓冲区中的完整think标签提取到推理内容: {extracted[:50]}...")
                return True, extracted
                
        # 如果只有开始标签但累积了足够的内容，也进行输出
        elif len(buffer) > 1000 or buffer.count("\n") > 3:
            start = buffer.find("<think>") + len("<think>")
            extracted = buffer[start:].strip()
            # 保留部分缓冲区，以便后续检测结束标签
            self._content_buffer = buffer[-100:] if len(buffer) > 100 else buffer
            logger.debug(f"从缓冲区中的不完整think标签提取到推理内容: {extracted[:50]}...")
            return True, extracted
            
        return False, ""
        
    def _extract_from_think_tags(self, text: str) -> tuple[bool, str]:
        """从think标签中提取推理内容"""
        if not text or "<think>" not in text:
            return False, ""
            
        if "</think>" in text:
            # 完整标签
            start = text.find("<think>") + len("<think>")
            end = text.find("</think>")
            if start < end:
                extracted = text[start:end].strip()
                logger.debug(f"从完整think标签中提取到推理内容: {extracted[:50]}...")
                return True, extracted
        else:
            # 不完整标签，只有开始
            start = text.find("<think>") + len("<think>")
            if start < len(text):
                extracted = text[start:].strip()
                logger.debug(f"从不完整think标签中提取到推理内容: {extracted[:50]}...")
                return True, extracted
                
        return False, ""
        
    def _extract_reasoning_from_text(self, text: str) -> tuple[bool, str]:
        """从文本中提取推理内容 (兼容旧版接口)"""
        return self._extract_from_think_tags(text)

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
        
        # 重置会话状态
        self._content_buffer = ""
        self._reasoning_buffer = ""
        self._has_found_reasoning = False
        self._content_token_count = 0
        
        try:
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode('utf-8')
                if not chunk_str.strip():
                    continue
                    
                try:
                    # 处理SSE格式: 'data: {"id":"..."}'
                    if chunk_str.startswith('data:'):
                        chunk_str = chunk_str[5:].strip()
                        if chunk_str == "[DONE]":
                            continue
                            
                    data = json.loads(chunk_str)
                    if not data or not data.get("choices") or not data["choices"][0].get("delta"):
                        continue
                        
                    delta = data["choices"][0]["delta"]
                    
                    # 使用 _extract_reasoning 提取推理内容
                    has_reasoning, reasoning = self._extract_reasoning(delta)
                    if has_reasoning and reasoning:
                        logger.debug(f"收到推理内容: {reasoning[:min(30, len(reasoning))]}...")
                        self._reasoning_buffer += reasoning  # 累积推理内容
                        yield "reasoning", reasoning
                    # 如果delta中有content但没有推理内容，则输出content
                    elif "content" in delta and delta["content"]:
                        content = delta["content"]
                        logger.debug(f"收到回答内容: {content[:min(30, len(content))]}...")
                        yield "content", content
                        
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析错误: {chunk_str[:50]}...")
                    continue
                    
        except Exception as e:
            logger.error(f"流式对话发生错误: {e}", exc_info=True)
            raise
            
    async def get_reasoning(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取推理过程
        
        根据配置使用不同的推理提取方式:
        1. 原始推理格式: 通过 reasoning_content 字段获取
        2. 标签格式: 通过 <think></think> 标签获取
        
        优化版本增强了内容处理的健壮性和错误恢复能力
        """
        model_arg = kwargs.get('model_arg')
        
        # 构建请求头和数据
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        # 验证并确保使用兼容的模型名称
        supported_models = {
            'deepseek': ['deepseek-reasoner'],
            'siliconflow': ['deepseek-ai/DeepSeek-R1'],
            'nvidia': ['deepseek-ai/deepseek-r1']
        }
        
        if self.provider in supported_models and model not in supported_models[self.provider]:
            logger.warning(f"请求的模型 '{model}' 可能不被 {self.provider} 提供商支持，将使用默认模型")
            model = supported_models[self.provider][0]  # 使用提供商的默认模型
        
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
        
        logger.info(f"开始获取推理内容，模型: {model}，提供商: {self.provider}，推理模式: {self.reasoning_mode}")
        logger.debug(f"推理请求数据: {data}")
        
        # 重置会话状态
        self._content_buffer = ""
        self._reasoning_buffer = ""
        self._has_found_reasoning = False
        self._content_token_count = 0
        
        # 用于拼接不完整的JSON
        buffer = ""
        has_yielded_content = False
        is_first_chunk = True
        
        try:
            async for chunk in self._make_request(headers, data):
                try:
                    chunk_str = chunk.decode('utf-8')
                    if not chunk_str.strip():
                        continue
                        
                    # 记录首个响应块，帮助诊断API格式问题
                    if is_first_chunk:
                        logger.debug(f"首个响应块: {chunk_str}")
                        is_first_chunk = False
                        
                    # 处理多行SSE数据
                    for line in chunk_str.splitlines():
                        if not line.strip():
                            continue
                            
                        if line.startswith("data: "):
                            json_str = line[len("data: "):].strip()
                            if json_str == "[DONE]":
                                logger.debug("收到[DONE]标记")
                                continue
                            
                            try:
                                # 解析JSON数据
                                data = json.loads(json_str)
                                if logger.isEnabledFor(logging.DEBUG):
                                    small_data = {k: v for k, v in data.items() if k != 'choices'}
                                    if 'choices' in data and data['choices']:
                                        small_data['choices_count'] = len(data['choices'])
                                        small_data['sample_delta'] = data['choices'][0].get('delta', {})
                                    logger.debug(f"解析JSON响应: {small_data}")
                                
                                if not data or not data.get("choices") or not data["choices"][0].get("delta"):
                                    logger.debug(f"跳过无效数据块: {json_str[:50]}")
                                    continue
                                
                                delta = data["choices"][0]["delta"]
                                
                                # 使用增强的_extract_reasoning方法处理
                                has_reasoning, reasoning = self._extract_reasoning(delta)
                                if has_reasoning and reasoning:
                                    logger.debug(f"获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
                                    self._reasoning_buffer += reasoning  # 累积推理内容
                                    yield "reasoning", reasoning
                                    has_yielded_content = True
                                # 如果有content但没有推理内容，作为常规内容输出
                                elif "content" in delta and delta["content"]:
                                    content = delta["content"]
                                    logger.debug(f"获取到普通内容: {content[:min(30, len(content))]}...")
                                    yield "content", content
                                    has_yielded_content = True
                                else:
                                    logger.debug(f"无法提取内容，delta: {delta}")
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON解析错误: {e}, 内容: {json_str[:50]}...")
                                # 可能是不完整的JSON，添加到缓冲区
                                buffer += json_str
                                try:
                                    data = json.loads(buffer)
                                    logger.debug(f"从缓冲区解析JSON成功")
                                    buffer = ""  # 重置缓冲区
                                    
                                    if data and data.get("choices") and data["choices"][0].get("delta"):
                                        delta = data["choices"][0]["delta"]
                                        
                                        has_reasoning, reasoning = self._extract_reasoning(delta)
                                        if has_reasoning and reasoning:
                                            logger.debug(f"从缓冲区获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
                                            self._reasoning_buffer += reasoning  # 累积推理内容
                                            yield "reasoning", reasoning
                                            has_yielded_content = True
                                        elif "content" in delta and delta["content"]:
                                            content = delta["content"]
                                            logger.debug(f"从缓冲区获取到普通内容: {content[:min(30, len(content))]}...")
                                            yield "content", content
                                            has_yielded_content = True
                                except Exception as e:
                                    # 仍然不是有效的JSON，继续等待更多数据
                                    logger.debug(f"缓冲区JSON解析失败: {e}")
                
                except Exception as e:
                    logger.warning(f"处理推理内容块时发生错误: {e}")
                    continue
            
            # 尝试从内容缓冲区中提取最后的推理内容
            if not has_yielded_content and self._content_buffer:
                logger.info(f"尝试从内容缓冲区中提取推理内容，缓冲区大小: {len(self._content_buffer)}")
                has_reasoning, reasoning = self._extract_from_buffered_think_tags()
                if has_reasoning and reasoning:
                    logger.debug(f"从最终缓冲区获取到推理内容: {reasoning[:min(30, len(reasoning))]}...")
                    yield "reasoning", reasoning
                    has_yielded_content = True
                elif self.reasoning_mode in ['auto', 'any_content', 'early_content']:
                    # 如果是这些模式，将剩余缓冲区内容作为推理输出
                    logger.debug(f"将剩余缓冲区内容作为推理输出")
                    yield "reasoning", self._content_buffer
                    has_yielded_content = True
            
            if not has_yielded_content:
                logger.warning("未能获取到任何推理内容或普通内容，请检查API响应格式")
                logger.warning(f"已尝试的推理模式: {self.reasoning_mode}")
                logger.warning(f"缓冲区状态: 内容缓冲区长度={len(self._content_buffer)}, 推理缓冲区长度={len(self._reasoning_buffer)}")
                
        except Exception as e:
            logger.error(f"获取推理内容过程中发生错误: {e}", exc_info=True)
            raise
