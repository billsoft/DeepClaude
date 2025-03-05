from .base import BaseReasoningProvider
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
import os
import json
import aiohttp
from app.utils.logger import logger

class DeepSeekReasoningProvider(BaseReasoningProvider):
    """基于DeepSeek的推理提供者"""
    
    def __init__(self, api_key: str, api_url: str = None, provider: str = "deepseek"):
        """初始化DeepSeek推理提供者
        
        Args:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API地址，如果为None则使用默认地址
            provider: 提供商类型，支持deepseek/siliconflow/nvidia
        """
        super().__init__(api_key, api_url)
        self.provider = provider.lower()
        
        # 设置默认API地址
        if not self.api_url:
            if self.provider == "deepseek":
                self.api_url = "https://api.deepseek.com/v1/chat/completions"
            elif self.provider == "siliconflow":
                self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
            elif self.provider == "nvidia":
                self.api_url = "https://api.nvidia.com/v1/chat/completions"
            else:
                raise ValueError(f"不支持的提供商: {provider}")
                
        # 推理模式配置
        self.reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto')
        logger.info(f"初始化DeepSeek推理提供者: provider={self.provider}, url={self.api_url}, mode={self.reasoning_mode}")
        
    async def extract_reasoning_from_think_tags(self, content: str) -> str:
        """从<think>标签中提取推理内容
        
        Args:
            content: 包含<think>标签的内容
            
        Returns:
            提取的推理内容
        """
        if "<think>" in content and "</think>" in content:
            start = content.find("<think>") + 7
            end = content.find("</think>")
            if start < end:
                return content[start:end].strip()
        return ""
        
    async def get_reasoning(self, messages: List[Dict], model: str = None, model_arg: tuple = None, **kwargs) -> str:
        """获取DeepSeek推理内容
        
        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认值
            model_arg: 模型参数(temperature, top_p)
            **kwargs: 其他参数
            
        Returns:
            推理内容
        """
        temperature = model_arg[0] if model_arg and len(model_arg) > 0 else kwargs.get('temperature', 0.7)
        top_p = model_arg[1] if model_arg and len(model_arg) > 1 else kwargs.get('top_p', 0.9)
        
        # 如果未指定模型，使用环境变量或默认值
        if not model:
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-R1')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": kwargs.get('max_tokens', 4096)
        }
        
        # 针对不同提供商的配置调整
        if self.provider == 'siliconflow':
            # 硅基流动可能有特殊参数
            if not data.get("stop"):
                data["stop"] = kwargs.get('stop', ["<STOP>"])
                
        elif self.provider == 'nvidia':
            # NVIDIA配置可能需要调整
            pass
            
        reasoning_content = []
        try:
            logger.info(f"发送DeepSeek推理请求: {self.api_url}")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求体: {json.dumps(data, ensure_ascii=False)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API请求失败: HTTP {response.status}\n{error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    logger.info("DeepSeek开始流式响应")
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if not line_str or not line_str.startswith('data:'):
                            continue
                        
                        data_json = line_str[5:].strip()
                        if data_json == "[DONE]":
                            logger.debug("收到[DONE]标记")
                            continue
                            
                        try:
                            data = json.loads(data_json)
                            if not data.get("choices"):
                                continue
                                
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            
                            # 根据推理模式提取内容
                            if self.reasoning_mode == 'reasoning_field':
                                # 直接从专用字段获取
                                reasoning = choice.get("reasoning_content")
                                if reasoning:
                                    reasoning_content.append(reasoning)
                                    
                            elif self.reasoning_mode == 'think_tags':
                                # 从<think>标签提取
                                content = delta.get("content", "")
                                if "<think>" in content:
                                    reasoning = await self.extract_reasoning_from_think_tags(content)
                                    if reasoning:
                                        reasoning_content.append(reasoning)
                                        
                            else:  # auto或者content模式
                                # 直接使用全部内容
                                content = delta.get("content", "")
                                if content:
                                    reasoning_content.append(content)
                                    
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {e}, 数据: {data_json[:100]}")
                            continue
                        except Exception as e:
                            logger.error(f"处理推理响应时出错: {e}")
                            continue
                            
            full_reasoning = "".join(reasoning_content)
            logger.info(f"获取到推理内容: {len(full_reasoning)} 字符")
            logger.debug(f"推理内容预览: {full_reasoning[:200]}...")
            return full_reasoning
            
        except Exception as e:
            logger.error(f"获取推理内容失败: {e}", exc_info=True)
            return f"获取推理内容时出错: {str(e)}" 