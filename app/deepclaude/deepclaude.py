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
    def __init__(
        self,
        deepseek_api_key: str,
        claude_api_key: str,
        deepseek_api_url: str = None,
        claude_api_url: str = None,
        claude_provider: str = None,
        is_origin_reasoning: bool = None,
        ollama_api_url: str = None
    ):
        # 验证必要的配置
        if not deepseek_api_key and os.getenv('REASONING_PROVIDER') == 'deepseek':
            raise ValueError("使用 DeepSeek 推理时必须提供 DEEPSEEK_API_KEY")
        
        if not claude_api_key:
            raise ValueError("必须提供 CLAUDE_API_KEY")
        
        # 验证 Ollama 配置
        if os.getenv('REASONING_PROVIDER') == 'ollama':
            ollama_url = ollama_api_url or os.getenv('OLLAMA_API_URL')
            if not ollama_url:
                raise ValueError("使用 Ollama 推理时必须提供 OLLAMA_API_URL")
        
        # 初始化推理提供者
        self.reasoning_providers = {
            'deepseek': lambda: DeepSeekClient(deepseek_api_key, deepseek_api_url),
            'ollama': lambda: OllamaR1Client(ollama_api_url)
        }
        
        # 初始化 Claude 客户端
        self.claude_client = ClaudeClient(claude_api_key, claude_api_url, claude_provider)
        
        self.is_origin_reasoning = (
            is_origin_reasoning 
            if is_origin_reasoning is not None 
            else os.getenv('IS_ORIGIN_REASONING', 'true').lower() == 'true'
        )

    def _get_reasoning_provider(self):
        """获取当前配置的推理提供者"""
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

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = None,
        claude_model: str = None
    ) -> AsyncGenerator[bytes, None]:
        """处理流式对话"""
        self._validate_messages(messages)
        deepseek_model = deepseek_model or os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-R1')
        claude_model = claude_model or os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
        
        # 添加模型名称验证
        self._validate_model_names(deepseek_model, claude_model)
        
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        response_queue = asyncio.Queue()
        
        try:
            provider = self._get_reasoning_provider()
            model = deepseek_model if isinstance(provider, DeepSeekClient) else (
                "deepseek-r1:32b" if isinstance(provider, OllamaR1Client) else claude_model
            )
            
            kwargs = {
                "messages": messages,
                "model": model,
            }
            
            if not isinstance(provider, OllamaR1Client):
                kwargs["model_arg"] = model_arg
            
            if isinstance(provider, DeepSeekClient):
                kwargs["is_origin_reasoning"] = self.is_origin_reasoning
            
            reasoning_content = []
            
            # 1. 开始思考提示
            yield self._format_stream_response(
                "🤔 思考过程:\n",
                chat_id,
                created_time,
                model
            )
            
            # 2. 获取并实时显示推理过程
            current_reasoning = ""
            try:
                async for content_type, content in provider.stream_chat(**kwargs):
                    if content_type == "reasoning":
                        # 收集完整推理内容用于后续 Claude
                        reasoning_content.append(content)
                        
                        # 处理增量内容，实现字符级流式输出
                        for char in content:
                            yield self._format_stream_response(
                                char,
                                chat_id,
                                created_time,
                                model
                            )
                            # 适当延迟，避免输出太快
                            await asyncio.sleep(0.01)
                        
                        # 每段推理后添加换行
                        yield self._format_stream_response(
                            "\n",
                            chat_id,
                            created_time,
                            model
                        )
                        
            except Exception as e:
                logger.error(f"获取推理内容失败: {e}")
                yield self._format_stream_response(
                    "\n❌ 思考过程获取失败，请稍后重试\n",
                    chat_id,
                    created_time,
                    model
                )
                return
            
            # 3. 分隔符
            yield self._format_stream_response(
                "\n=============== 思考完毕，开始回答 ===============\n\n",
                chat_id,
                created_time,
                model
            )
            
            # 4. 调用 Claude 并实时显示回答
            try:
                # 构造 Claude 输入
                reasoning = "\n".join(reasoning_content)
                combined_content = f"""
这是我自己基于问题的思考过程:\n{reasoning}\n\n
上面是我自己的思考过程不一定完全正确请借鉴思考过程和期中你也认为正确的部分（1000% 权重）
，现在请给出详细和细致的答案，不要省略步骤和步骤细节
，要分解原题确保你理解了原题的每个部分，也要掌握整体意思
，最佳质量（1000% 权重），最详细解答（1000% 权重），不要回答太简单让我能参考一步步应用（1000% 权重）:"""

                claude_messages = [{"role": "user", "content": combined_content}]
                
                # 字符级流式输出 Claude 回答
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model_arg=model_arg,
                    model=claude_model
                ):
                    if content_type == "answer":
                        # 一个字符一个字符地输出
                        for char in content:
                            yield self._format_stream_response(
                                char,
                                chat_id,
                                created_time,
                                model
                            )
                            # 适当延迟，避免输出太快
                            await asyncio.sleep(0.01)
                            
            except Exception as e:
                logger.error(f"获取 Claude 回答失败: {e}")
                yield self._format_stream_response(
                    "\n❌ 获取回答失败，请稍后重试\n",
                    chat_id,
                    created_time,
                    model
                )
            
        except Exception as e:
            logger.error(f"处理流式对话时发生错误: {e}")
            yield self._format_stream_response(
                "\n❌ 服务出现错误，请稍后重试\n",
                chat_id,
                created_time,
                model
            )
        finally:
            yield b'data: [DONE]\n\n'

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
            reasoning = await self._get_reasoning_with_fallback(
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

    async def _get_reasoning_content(self, messages: list, model: str, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
        """获取推理内容的统一接口
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称
            **kwargs: 额外参数
            
        Yields:
            tuple[str, str]: (content_type, content)
                - content_type: "reasoning" 表示推理过程，"content" 表示最终答案
                - content: 具体内容
                
        Raises:
            Exception: 当获取推理内容失败时抛出
        """
        provider = self._get_reasoning_provider()
        try:
            async for content_type, content in provider.get_reasoning(
                messages=messages,
                model=model,
                **kwargs
            ):
                yield content_type, content
        except Exception as e:
            logger.error(f"获取推理内容时发生错误: {e}", exc_info=True)
            raise

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

    def _format_stream_response(self, content: str, chat_id: str, created_time: int, model: str) -> bytes:
        """格式化流式响应"""
        response = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content
                }
            }]
        }
        return f"data: {json.dumps(response)}\n\n".encode('utf-8')