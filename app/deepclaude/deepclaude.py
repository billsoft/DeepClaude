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
        # 保存配置参数，用于验证
        self.deepseek_api_key = kwargs.get('deepseek_api_key')
        self.deepseek_api_url = kwargs.get('deepseek_api_url')
        self.ollama_api_url = kwargs.get('ollama_api_url')
        
        # 1. 初始化回答者(Claude)
        self.claude_client = ClaudeClient(
            api_key=kwargs.get('claude_api_key'),
            api_url=kwargs.get('claude_api_url'),
            provider=kwargs.get('claude_provider')
        )
        
        # 保存provider属性，用于模型选择
        self.provider = kwargs.get('deepseek_provider', 'deepseek')
        
        # 2. 配置思考者映射
        self.reasoning_providers = {
            'deepseek': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key'),
                api_url=kwargs.get('deepseek_api_url'),
                provider=kwargs.get('deepseek_provider')
            ),
            'ollama': lambda: OllamaR1Client(
                api_url=kwargs.get('ollama_api_url')
            ),
            'siliconflow': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key'),
                api_url=kwargs.get('deepseek_api_url'),
                provider='siliconflow'
            ),
            'nvidia': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key'),
                api_url=kwargs.get('deepseek_api_url'),
                provider='nvidia'
            )
        }
        
        # 3. 推理提取配置
        self.min_reasoning_chars = int(os.getenv('MIN_REASONING_CHARS', '50'))
        self.max_retries = int(os.getenv('REASONING_MAX_RETRIES', '2'))
        self.reasoning_modes = os.getenv('REASONING_MODE_SEQUENCE', 'auto,think_tags,early_content,any_content').split(',')

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
                        "is_reasoning": True,  # 添加顶层标记
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"🤔 思考过程:\n{content}\n",
                                "reasoning": True  # 在delta中添加标记
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

    async def _handle_api_error(self, e: Exception) -> str:
        """处理 API 错误"""
        if isinstance(e, aiohttp.ClientError):
            return "网络连接错误，请检查网络连接"
        elif isinstance(e, asyncio.TimeoutError):
            return "请求超时，请稍后重试"
        elif isinstance(e, ValueError):
            return f"参数错误: {str(e)}"
        else:
            return f"未知错误: {str(e)}"

    async def chat_completions_with_stream(self, messages: list, **kwargs):
        try:
            logger.info("开始流式处理请求...")
            
            # 获取思考者实例
            provider = self._get_reasoning_provider()
            
            # 用于收集思考内容
            reasoning_content = []
            thought_complete = False
            
            # 记录参数信息，便于调试
            logger.info(f"思考者提供商: {self.provider}")
            logger.info(f"思考模式: {os.getenv('DEEPSEEK_REASONING_MODE', 'auto')}")
            
            # 1. 思考阶段 - 直接转发 token 并收集内容
            try:
                # 获取推理内容并设置重试逻辑
                reasoning_success = False
                is_first_reasoning = True  # 新增标记，表示是否是首次发送思考内容
                
                # 首先向前端发送开始思考的提示
                yield self._format_stream_response(
                    "开始思考问题...",
                    content_type="reasoning",
                    is_first_thought=True,  # 标记这是首个思考内容
                    **kwargs
                )
                is_first_reasoning = False  # 发送完首个提示后设为False
                
                # 遍历不同的推理模式直到成功
                for retry_count, reasoning_mode in enumerate(self.reasoning_modes):
                    if reasoning_success:
                        logger.info("推理成功，退出模式重试循环")
                        break
                        
                    # 如果思考完成且已收集足够推理内容，直接进入回答阶段
                    if thought_complete and len("".join(reasoning_content)) > self.min_reasoning_chars:
                        logger.info("思考阶段已完成，退出所有重试")
                        reasoning_success = True
                        break
                        
                    if retry_count > 0:
                        logger.info(f"尝试使用不同的推理模式: {reasoning_mode} (尝试 {retry_count+1}/{len(self.reasoning_modes)})")
                        # 设置环境变量以更改推理模式
                        os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
                        # 重新初始化提供者，以加载新的推理模式
                        provider = self._get_reasoning_provider()
                        
                        # 通知前端正在切换模式
                        yield self._format_stream_response(
                            f"切换思考模式: {reasoning_mode}...",
                            content_type="reasoning",
                            is_first_thought=False,  # 非首次思考内容
                            **kwargs
                        )
                
                    # 准备思考参数
                    thinking_kwargs = self._prepare_thinker_kwargs(kwargs)
                    logger.info(f"使用思考模型: {thinking_kwargs.get('model')}")
                    
                    # 获取推理内容
                    try:
                        async for content_type, content in provider.get_reasoning(
                            messages=messages,
                            **thinking_kwargs
                        ):
                            if content_type == "reasoning":
                                # 保存思考内容
                                reasoning_content.append(content)
                                # 如果收集了足够多的推理内容，标记为成功
                                if len("".join(reasoning_content)) > self.min_reasoning_chars:
                                    reasoning_success = True
                                # 直接转发思考 token，明确标记为推理内容
                                yield self._format_stream_response(
                                    content, 
                                    content_type="reasoning",
                                    is_first_thought=False,  # 非首次思考内容
                                    **kwargs
                                )
                            elif content_type == "content":
                                # 如果收到常规内容，说明思考阶段可能已结束
                                logger.debug(f"收到常规内容: {content[:50]}...")
                                thought_complete = True
                                
                                # 如果还没有足够的推理内容，但收到了常规内容，可以将其转化为推理内容
                                if not reasoning_success and reasoning_mode in ['early_content', 'any_content']:
                                    logger.info("将常规内容转化为推理内容")
                                    reasoning_content.append(f"分析: {content}")
                                    yield self._format_stream_response(
                                        f"分析: {content}", 
                                        content_type="reasoning",
                                        is_first_thought=False,  # 非首次思考内容
                                        **kwargs
                                    )
                                    
                                # 重要: 如果已收集足够的推理内容或处于特定模式，则退出循环
                                if len("".join(reasoning_content)) > self.min_reasoning_chars or reasoning_mode in ['early_content', 'any_content']:
                                    logger.info("收到常规内容且已收集足够推理内容，终止推理过程")
                                    reasoning_success = True
                                    break
                    except Exception as reasoning_e:
                        logger.error(f"使用模式 {reasoning_mode} 获取推理内容时发生错误: {reasoning_e}")
                        # 通知前端当前模式失败
                        yield self._format_stream_response(
                            f"思考模式 {reasoning_mode} 失败，尝试其他方式...",
                            content_type="reasoning",
                            is_first_thought=False,  # 非首次思考内容
                            **kwargs
                        )
                        continue
                
                logger.info(f"思考过程{'成功' if reasoning_success else '失败'}，共收集 {len(reasoning_content)} 个思考片段")
            except Exception as e:
                logger.error(f"思考阶段发生错误: {e}", exc_info=True)
                # 记录错误但继续尝试使用已收集的内容
                yield self._format_stream_response(
                    f"思考过程出错: {str(e)}，尝试继续...",
                    content_type="reasoning",
                    is_first_thought=False,  # 非首次思考内容
                    **kwargs
                )
                
            # 确保思考内容不为空且有足够的内容
            if not reasoning_content or len("".join(reasoning_content)) < self.min_reasoning_chars:
                logger.warning(f"未获取到足够的思考内容，当前内容长度: {len(''.join(reasoning_content))}")
                
                # 如果接近但不满足最小需求，仍然使用它
                if not reasoning_content or len("".join(reasoning_content)) < self.min_reasoning_chars // 2:
                    logger.warning("未获取到有效思考内容，使用原始问题作为替代")
                    message_content = messages[-1]['content'] if messages and isinstance(messages[-1], dict) and 'content' in messages[-1] else "未能获取问题内容"
                    reasoning_content = [f"问题分析：{message_content}"]
                    # 也向用户发送提示，明确标记为推理内容
                    yield self._format_stream_response(
                        "无法获取思考过程，将直接回答问题",
                        content_type="reasoning",
                        is_first_thought=True,  # 这是新的思考过程的开始
                        **kwargs
                    )
            
            # 进入回答阶段前发送分隔符
            yield self._format_stream_response(
                "\n\n---\n思考完毕，开始回答：\n\n",
                content_type="separator",
                is_first_thought=False,  # 非思考内容
                **kwargs
            )
            
            # 2. 回答阶段 - 使用格式化的 prompt 并转发 token
            full_reasoning = "\n".join(reasoning_content)
            if 'content' in messages[-1]:
                original_question = messages[-1]['content']
            else:
                logger.warning("无法从消息中获取问题内容")
                original_question = "未提供问题内容"
                
            prompt = self._format_claude_prompt(
                original_question,
                full_reasoning
            )
            
            logger.debug(f"发送给Claude的提示词: {prompt[:500]}...")
            
            try:
                answer_begun = False
                async for content_type, content in self.claude_client.stream_chat(
                    messages=[{"role": "user", "content": prompt}],
                    **self._prepare_answerer_kwargs(kwargs)
                ):
                    if content_type == "content" and content:
                        if not answer_begun and content.strip():
                            # 标记回答开始
                            answer_begun = True
                            
                        # 转发回答 token，明确标记为普通内容
                        yield self._format_stream_response(
                            content,
                            content_type="content",
                            is_first_thought=False,  # 非思考内容
                            **kwargs
                        )
            except Exception as e:
                logger.error(f"回答阶段发生错误: {e}", exc_info=True)
                yield self._format_stream_response(
                    f"\n\n⚠️ 获取回答时发生错误: {str(e)}",
                    content_type="error",
                    is_first_thought=False,  # 非思考内容
                    **kwargs
                )
                
        except Exception as e:
            error_msg = await self._handle_api_error(e)
            logger.error(f"流式处理错误: {error_msg}", exc_info=True)
            yield self._format_stream_response(
                f"错误: {error_msg}", 
                content_type="error",
                is_first_thought=False,  # 非思考内容
                **kwargs
            )

    def _prepare_thinker_kwargs(self, kwargs: dict) -> dict:
        """准备思考者参数"""
        provider_type = os.getenv('REASONING_PROVIDER', 'deepseek').lower()
        
        if provider_type == 'ollama':
            model = "deepseek-r1:32b"
        else:
            # 不再使用kwargs中传入的model参数，以避免使用不兼容的模型名称
            # 而是使用环境变量或默认的DeepSeek模型名称
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')
            
            # 根据provider进行特定处理
            if self.provider == 'deepseek':
                model = 'deepseek-reasoner'  # 使用确定可用的模型
            elif self.provider == 'siliconflow':
                model = 'deepseek-ai/DeepSeek-R1'
            elif self.provider == 'nvidia':
                model = 'deepseek-ai/deepseek-r1'
            
        return {
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9)
        }
        
    def _prepare_answerer_kwargs(self, kwargs: dict) -> dict:
        """准备回答者参数"""
        return {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9)
        }

    def _chunk_content(self, content: str, chunk_size: int = 3) -> list[str]:
        """将内容分割成小块以实现更细粒度的流式输出
        
        Args:
            content: 要分割的内容
            chunk_size: 每个块的大小，默认为3个字符
            
        Returns:
            list[str]: 分割后的内容块列表
        """
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

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
4. 如果思考过程有错误或不完整，请指出并补充正确的解答
5. 保持回答的专业性和准确性
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
            
            # 尝试使用不同的推理模式重试
            for reasoning_mode in self.reasoning_modes[1:]:  # 跳过第一个已使用的模式
                try:
                    logger.info(f"尝试使用不同的推理模式获取内容: {reasoning_mode}")
                    os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
                    reasoning = await self._get_reasoning_content(
                        messages=messages,
                        model=deepseek_model,
                        model_arg=model_arg
                    )
                    if reasoning and len(reasoning) > self.min_reasoning_chars:
                        logger.info(f"使用推理模式 {reasoning_mode} 成功获取推理内容")
                        break
                except Exception as retry_e:
                    logger.error(f"使用推理模式 {reasoning_mode} 重试失败: {retry_e}")
        
        logger.debug(f"获取到推理内容: {reasoning[:min(500, len(reasoning))]}...")
        
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
            full_content = ""
            async for content_type, content in self.claude_client.stream_chat(
                messages=claude_messages,
                model_arg=model_arg,
                model=claude_model,
                stream=False
            ):
                if content_type in ["answer", "content"]:
                    logger.debug(f"获取到 Claude 回答: {content}")
                    full_content += content
            
            # 返回完整的回答内容
            return {
                "content": full_content,
                "role": "assistant"
            }
        except Exception as e:
            logger.error(f"获取 Claude 回答失败: {e}")
            raise

    async def _get_reasoning_content(self, messages: list, model: str, **kwargs) -> str:
        """获取推理内容
        
        1. 首先尝试使用配置的推理提供者
        2. 如果失败则尝试切换到备用提供者
        3. 支持多种推理模式重试
        """
        try:
            provider = self._get_reasoning_provider()
            reasoning_content = []
            content_received = False
            
            logger.info(f"开始获取思考内容，模型: {model}, 推理模式: {os.getenv('DEEPSEEK_REASONING_MODE', 'auto')}")
            
            async for content_type, content in provider.get_reasoning(
                messages=messages,
                model=model,
                model_arg=kwargs.get('model_arg')  # 只传递必要的参数
            ):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                    logger.debug(f"收到推理内容，当前长度: {len(''.join(reasoning_content))}")
                elif content_type == "content" and not reasoning_content:
                    # 如果没有收集到推理内容，但收到了内容，将其也视为推理
                    logger.info("未收集到推理内容，将普通内容视为推理")
                    reasoning_content.append(f"分析: {content}")
                    logger.debug(f"普通内容转为推理内容，当前长度: {len(''.join(reasoning_content))}")
                elif content_type == "content":
                    # 记录收到普通内容，这通常表示推理阶段结束
                    content_received = True
                    logger.info("收到普通内容，推理阶段可能已结束")
                
            result = "\n".join(reasoning_content)
            
            # 如果已收到普通内容且推理内容长度足够，直接返回
            if content_received and len(result) > self.min_reasoning_chars:
                logger.info(f"已收到普通内容且推理内容长度足够 ({len(result)}字符)，结束获取推理")
                return result
            
            # 如果内容不足，尝试切换模式重试
            if not result or len(result) < self.min_reasoning_chars:
                current_mode = os.getenv('DEEPSEEK_REASONING_MODE', 'auto')
                logger.warning(f"使用模式 {current_mode} 获取的推理内容不足，尝试切换模式")
                
                # 尝试下一个推理模式
                for reasoning_mode in self.reasoning_modes:
                    if reasoning_mode == current_mode:
                        continue
                    
                    logger.info(f"尝试使用推理模式: {reasoning_mode}")
                    os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
                    provider = self._get_reasoning_provider()  # 重新初始化提供者
                    
                    reasoning_content = []
                    async for content_type, content in provider.get_reasoning(
                        messages=messages,
                        model=model,
                        model_arg=kwargs.get('model_arg')
                    ):
                        if content_type == "reasoning":
                            reasoning_content.append(content)
                        elif content_type == "content" and not reasoning_content:
                            reasoning_content.append(f"分析: {content}")
                    
                    retry_result = "\n".join(reasoning_content)
                    if retry_result and len(retry_result) > self.min_reasoning_chars:
                        logger.info(f"使用推理模式 {reasoning_mode} 成功获取足够的推理内容")
                        return retry_result
            
            return result or "无法获取足够的推理内容"
        except Exception as e:
            logger.error(f"主要推理提供者失败: {e}")
            if isinstance(provider, DeepSeekClient):
                # 获取当前提供商名称
                current_provider = getattr(provider, 'provider', 'unknown')
                logger.info(f"从 {current_provider} 提供商切换到 Ollama 推理提供者")
                try:
                    provider = OllamaR1Client(api_url=os.getenv('OLLAMA_API_URL'))
                    # 重试使用 Ollama
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
            
            # 尝试不同的推理模式
            for reasoning_mode in self.reasoning_modes:
                if reasoning_content and len("".join(reasoning_content)) > self.min_reasoning_chars:
                    # 如果已经收集到足够的内容，结束循环
                    logger.info(f"已收集到足够推理内容 ({len(''.join(reasoning_content))}字符)，不再尝试其他模式")
                    break
                    
                logger.info(f"尝试使用推理模式: {reasoning_mode}")
                os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
                provider = self._get_reasoning_provider()  # 重新初始化提供者
                
                temp_content = []
                content_received = False  # 标记是否收到普通内容
                
                try:
                    async for content_type, content in provider.get_reasoning(
                        messages=messages,
                        model=model,
                        model_arg=model_arg
                    ):
                        if content_type == "reasoning":
                            temp_content.append(content)
                            logger.debug(f"收到推理内容，当前临时内容长度: {len(''.join(temp_content))}")
                        elif content_type == "content" and not temp_content and reasoning_mode in ['early_content', 'any_content']:
                            # 在某些模式下，也将普通内容视为推理
                            temp_content.append(f"分析: {content}")
                            logger.debug(f"普通内容转为推理内容，当前临时内容长度: {len(''.join(temp_content))}")
                        elif content_type == "content":
                            # 收到普通内容，可能表示推理阶段结束
                            content_received = True
                            logger.info("收到普通内容，推理阶段可能已结束")
                            
                        # 如果收到普通内容且已有足够推理内容，提前终止
                        if content_received and len("".join(temp_content)) > self.min_reasoning_chars:
                            logger.info("收到普通内容且临时推理内容足够，提前结束推理获取")
                            break
                            
                    if temp_content and len("".join(temp_content)) > len("".join(reasoning_content)):
                        # 如果本次获取的内容更多，则更新结果
                        reasoning_content = temp_content
                        if content_received:
                            # 如果已收到普通内容，表示推理阶段已完成，不再尝试其他模式
                            logger.info("推理阶段已结束且内容足够，停止尝试其他模式")
                            break
                except Exception as mode_e:
                    logger.error(f"使用推理模式 {reasoning_mode} 时发生错误: {mode_e}")
                    continue
            
            return "".join(reasoning_content) or "无法获取推理内容"
        except Exception as e:
            logger.error(f"主要推理提供者失败: {e}")
            if isinstance(provider, DeepSeekClient):
                # 获取当前提供商名称
                current_provider = getattr(provider, 'provider', 'unknown')
                logger.info(f"从 {current_provider} 提供商切换到 Ollama 推理提供者")
                try:
                    provider = OllamaR1Client(api_url=os.getenv('OLLAMA_API_URL'))
                    # 重试使用 Ollama
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
        elif provider == 'siliconflow':
            if not self.deepseek_api_key:
                raise ValueError("使用 硅基流动 时必须提供 DeepSeek API KEY")
            if not self.deepseek_api_url:
                raise ValueError("使用 硅基流动 时必须提供 DeepSeek API URL")
        elif provider == 'nvidia':
            if not self.deepseek_api_key:
                raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API KEY")
            if not self.deepseek_api_url:
                raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API URL")

    def _format_stream_response(self, content: str, content_type: str = "content", **kwargs) -> bytes:
        """格式化流式响应
        
        Args:
            content: 要发送的内容
            content_type: 内容类型，可以是 "reasoning"、"content" 或 "separator"
            **kwargs: 其他参数
            
        Returns:
            bytes: 格式化的SSE响应
        """
        # 基本响应结构
        response = {
            "id": kwargs.get("chat_id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion.chunk",
            "created": kwargs.get("created_time", int(time.time())),
            "model": kwargs.get("model", "deepclaude"),
            "choices": [{
                "index": 0,
                "delta": {
                    "content": content
                }
            }]
        }
        
        # 为不同内容类型添加明显标记
        if content_type == "reasoning":
            # 添加思考标记 - 在delta中和response根级别都添加标记
            response["choices"][0]["delta"]["reasoning"] = True
            response["is_reasoning"] = True  # 根级别添加标记，方便前端识别
            
            # 只在首个token添加表情符号，后续token保持原样
            # 检查是否已经是以表情符号开头，如果不是，并且是首次发送思考内容(可从kwargs中获取标志)，则添加表情符号
            is_first_thought = kwargs.get("is_first_thought", False)
            if is_first_thought and not content.startswith("🤔"):
                response["choices"][0]["delta"]["content"] = f"🤔 {content}"
        elif content_type == "separator":
            # 分隔符特殊标记
            response["is_separator"] = True
        elif content_type == "error":
            # 错误信息特殊标记
            response["is_error"] = True
            response["choices"][0]["delta"]["content"] = f"⚠️ {content}"
        
        return f"data: {json.dumps(response)}\n\n".encode('utf-8')

    def _validate_kwargs(self, kwargs: dict) -> None:
        """验证参数的有效性"""
        # 验证温度参数
        temperature = kwargs.get('temperature')
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1:
                raise ValueError("temperature 必须在 0 到 1 之间")
            
        # 验证 top_p 参数
        top_p = kwargs.get('top_p')
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                raise ValueError("top_p 必须在 0 到 1 之间")
            
        # 验证模型参数
        model = kwargs.get('model')
        if model and not isinstance(model, str):
            raise ValueError("model 必须是字符串类型")

    def _split_into_tokens(self, text: str) -> list[str]:
        """将文本分割成更小的token
        
        Args:
            text: 要分割的文本
            
        Returns:
            list[str]: token列表
        """
        # 可以根据需要调整分割粒度
        # 1. 按字符分割
        return list(text)
        
        # 或者按词分割
        # return text.split()
        
        # 或者使用更复杂的分词算法
        # return some_tokenizer(text)