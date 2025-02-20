"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用

主要功能：
1. 集成 DeepSeek 和 Claude 两个大语言模型的能力
2. 支持流式和非流式两种输出模式
3. 实现 DeepSeek 推理结果作为 Claude 输入的串联调用
4. 提供符合 OpenAI API 格式的标准输出

工作流程：
1. 接收用户输入的消息和模型参数
2. 调用 DeepSeek 进行推理，获取推理过程
3. 将推理结果传递给 Claude 进行处理
4. 整合输出结果并返回给用户

技术特点：
1. 使用异步编程提高并发性能
2. 采用队列机制实现数据流转
3. 支持流式输出提升用户体验
4. 完善的错误处理和日志记录
"""
import json
import time
import tiktoken
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, OllamaR1Client
from app.utils.message_processor import MessageProcessor
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接
    
    该类负责协调 DeepSeek 和 Claude 两个模型的调用过程，主要特点：
    1. 支持流式和非流式两种输出模式
    2. 实现 DeepSeek 推理和 Claude 回答的串联调用
    3. 提供标准的 OpenAI 格式输出
    4. 支持多种 API 提供商配置
    
    主要组件：
    - DeepSeek客户端：负责调用 DeepSeek API 获取推理过程
    - Claude客户端：负责调用 Claude API 生成最终答案
    - 异步队列：用于数据流转和任务协调
    - OllamaR1Client：负责调用 OllamaR1 API 生成最终答案
    - 异步队列：用于数据流转和任务协调
    
    工作模式：
    1. 流式模式：实时返回推理过程和生成结果
    2. 非流式模式：等待完整结果后一次性返回
    """
    def __init__(self, **kwargs):
        """初始化 DeepClaude
        
        工作流程:
        1. 思考者(DeepSeek/Ollama)提供思考过程
        2. 回答者(Claude)根据思考过程和原题生成答案
        """
        # 1. 初始化回答者(Claude)
        self.claude_client = ClaudeClient(
            api_key=kwargs.get('claude_api_key'),
            api_url=kwargs.get('claude_api_url'),
            provider=kwargs.get('claude_provider')
        )
        
        # 2. 配置思考者映射
        self.reasoning_providers = {
            'deepseek': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key'),
                api_url=kwargs.get('deepseek_api_url'),
                provider=kwargs.get('deepseek_provider')
            ),
            'ollama': lambda: OllamaR1Client(
                api_url=kwargs.get('ollama_api_url')
            )
        }

    def _get_reasoning_provider(self):
        """获取思考者实例"""
        provider = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
        
        if provider not in self.reasoning_providers:
            raise ValueError(f"不支持的推理提供者: {provider}")
            
        return self.reasoning_providers[provider]()

    async def _handle_stream_response(self, response_queue: asyncio.Queue, 
                                    chat_id: str, created_time: int, model: str) -> AsyncGenerator[bytes, None]:
        """处理流式响应"""
        try:
            while True:
                item = await response_queue.get()
                if item is None:
                    break
                    
                content_type, content = item
                if content_type == "reasoning":
                    response = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"🤔 思考过程:\n{content}\n"
                            }
                        }]
                    }
                elif content_type == "content":
                    response = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"\n\n---\n思考完毕，开始回答：\n\n{content}"
                            }
                        }]
                    }
                yield f"data: {json.dumps(response)}\n\n".encode('utf-8')
        except Exception as e:
            logger.error(f"处理流式响应时发生错误: {e}", exc_info=True)
            raise

    async def chat_completions_with_stream(self, messages: list, model_arg: tuple = None, **kwargs):
        """流式对话完成"""
        try:
            # 1. 获取思考过程
            reasoning_content = []
            provider = self._get_reasoning_provider()
            
            # 根据提供者类型选择正确的模型和参数
            provider_type = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
            if provider_type == 'ollama':
                model = "deepseek-r1:32b"
                provider_kwargs = {}
            else:
                model = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')
                provider_kwargs = {'model_arg': model_arg} if model_arg else {}
            
            # 先输出思考标题
            yield self._format_stream_response("🤔 思考过程:\n", **kwargs)
            
            try:
                async for content_type, content in provider.get_reasoning(
                    messages=messages,
                    model=model,
                    **provider_kwargs
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        # 直接输出思考内容，不带标题
                        yield self._format_stream_response(content, **kwargs)
            except Exception as e:
                logger.error(f"获取思考过程失败: {e}")
            
            # 2. 构造给 Claude 的输入
            reasoning = "\n".join(reasoning_content) if reasoning_content else "无法获取思考过程"
            
            # 3. 使用 Claude 生成最终答案
            try:
                prompt = self._format_claude_prompt(messages[-1]['content'], reasoning)
                claude_messages = [{"role": "user", "content": prompt}]
                
                # 先输出回答标题
                yield self._format_stream_response("\n\n---\n最终答案：\n\n", **kwargs)
                
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model=os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
                    max_tokens=8192,
                    temperature=0.7,
                    top_p=0.9
                ):
                    if content_type == "answer":
                        # 直接输出回答内容，不带标题
                        yield self._format_stream_response(content, **kwargs)
            except Exception as e:
                logger.error(f"获取Claude回答失败: {e}")
                yield self._format_stream_response("❌ 获取最终答案失败，请稍后重试", **kwargs)
        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            yield self._format_stream_response("❌ 服务出现错误，请稍后重试", **kwargs)

    def _format_claude_prompt(self, original_question: str, reasoning: str) -> str:
        """格式化给Claude的提示词"""
        return f"""
原始问题:
{original_question}

思考过程:
{reasoning}

请基于以上思考过程和原始问题,给出详细的答案。要求:
1. 分步骤详细解答
2. 确保理解问题的每个部分
3. 给出完整的解决方案
"""

    async def chat_completions_without_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022"
    ) -> dict:
        """非流式对话完成
        
        Args:
            messages: 对话消息列表
            model_arg: 模型参数元组
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            
        Returns:
            dict: 包含回答内容的响应字典
        """
        logger.info("开始处理请求...")
        logger.debug(f"输入消息: {messages}")
        
        # 1. 获取推理内容
        logger.info("正在获取推理内容...")
        try:
            reasoning = await self._get_reasoning_content(
                messages=messages,
                model=deepseek_model,
                model_arg=model_arg
            )
        except Exception as e:
            logger.error(f"获取推理内容失败: {e}")
            reasoning = "无法获取推理内容"
        
        logger.debug(f"获取到推理内容: {reasoning}")
        
        # 2. 构造 Claude 的输入消息
        combined_content = f"""
这是我自己基于问题的思考过程:\n{reasoning}\n\n
上面是我自己的思考过程不一定完全正确请借鉴思考过程和期中你也认为正确的部分（1000% 权重）
，现在请给出详细和细致的答案，不要省略步骤和步骤细节
，要分解原题确保你理解了原题的每个部分，也要掌握整体意思
，最佳质量（1000% 权重），最详细解答（1000% 权重），不要回答太简单让我能参考一步步应用（1000% 权重）:"""
        
        claude_messages = [{"role": "user", "content": combined_content}]
        
        # 3. 获取 Claude 回答
        logger.info("正在获取 Claude 回答...")
        try:
            async for content_type, content in self.claude_client.stream_chat(
                messages=claude_messages,
                model_arg=model_arg,
                model=claude_model,
                stream=False
            ):
                if content_type == "answer":
                    logger.debug(f"获取到 Claude 回答: {content}")
                    return {
                        "content": content,
                        "role": "assistant"
                    }
        except Exception as e:
            logger.error(f"获取 Claude 回答失败: {e}")
            raise

    async def _get_reasoning_content(self, messages: list, model: str, **kwargs) -> str:
        """获取推理内容
        
        1. 首先尝试使用配置的推理提供者
        2. 如果失败则尝试切换到备用提供者
        """
        try:
            provider = self._get_reasoning_provider()
            reasoning_content = []
            
            async for content_type, content in provider.get_reasoning(
                messages=messages,
                model=model,
                model_arg=kwargs.get('model_arg')  # 只传递必要的参数
            ):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                
            return "\n".join(reasoning_content)
        except Exception as e:
            logger.error(f"主要推理提供者失败: {e}")
            # 如果配置了 Ollama 作为备用，则尝试切换
            if hasattr(self, 'ollama_api_url'):
                logger.info("尝试切换到 Ollama 推理提供者")
                try:
                    provider = OllamaR1Client(self.ollama_api_url)
                    reasoning_content = []
                    async for content_type, content in provider.get_reasoning(
                        messages=messages,
                        model="deepseek-r1:32b"
                    ):
                        if content_type == "reasoning":
                            reasoning_content.append(content)
                    return "\n".join(reasoning_content)
                except Exception as e:
                    logger.error(f"备用推理提供者也失败: {e}")
                
            return "无法获取推理内容"

    async def _retry_operation(self, operation, max_retries=3):
        """通用重试机制"""
        for i in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if i == max_retries - 1:
                    raise
                logger.warning(f"操作失败，正在重试 ({i+1}/{max_retries}): {str(e)}")
                await asyncio.sleep(1 * (i + 1))  # 指数退避

    def _validate_model_names(self, deepseek_model: str, claude_model: str):
        """验证模型名称的有效性"""
        if not deepseek_model or not isinstance(deepseek_model, str):
            raise ValueError("无效的 DeepSeek 模型名称")
        
        if not claude_model or not isinstance(claude_model, str):
            raise ValueError("无效的 Claude 模型名称")

    def _validate_messages(self, messages: list) -> None:
        """验证消息列表的有效性"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("消息必须是字典格式")
            if "role" not in msg or "content" not in msg:
                raise ValueError("消息必须包含 role 和 content 字段")
            if msg["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"不支持的消息角色: {msg['role']}")

    async def _get_reasoning_with_fallback(
        self, 
        messages: list,
        model: str,
        model_arg: tuple[float, float, float, float] = None
    ) -> str:
        """获取推理内容，带有备用方案
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称
            model_arg: 模型参数元组
            
        Returns:
            str: 推理内容
        """
        try:
            provider = self._get_reasoning_provider()
            reasoning_content = []
            
            async for content_type, content in provider.get_reasoning(
                messages=messages,
                model=model,
                model_arg=model_arg
            ):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                
            return "".join(reasoning_content)
        except Exception as e:
            logger.error(f"主要推理提供者失败: {e}")
            if isinstance(provider, DeepSeekClient):
                logger.info("尝试切换到 Ollama 推理提供者")
                provider = OllamaR1Client(self.ollama_api_url)
                # 重试使用 Ollama
                reasoning_content = []
                async for content_type, content in provider.get_reasoning(
                    messages=messages,
                    model="deepseek-r1:32b"
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                return "".join(reasoning_content)
            raise

    def _validate_config(self):
        """验证配置的完整性"""
        provider = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
        
        if provider == 'deepseek':
            if not self.deepseek_api_key:
                raise ValueError("使用 DeepSeek 时必须提供 API KEY")
            if not self.deepseek_api_url:
                raise ValueError("使用 DeepSeek 时必须提供 API URL")
        elif provider == 'ollama':
            if not self.ollama_api_url:
                raise ValueError("使用 Ollama 时必须提供 API URL")

    def _format_stream_response(self, content: str, **kwargs) -> bytes:
        """格式化流式响应为 OpenAI 格式"""
        if not content:  # 防止空内容
            return b""
        
        response = {
            "id": kwargs.get('chat_id', 'chatcmpl-default'),
            "object": "chat.completion.chunk",
            "created": kwargs.get('created_time', int(time.time())),
            "model": kwargs.get('model', 'deepclaude'),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(response)}\n\n".encode('utf-8')