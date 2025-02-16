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
from app.clients import DeepSeekClient, ClaudeClient
from app.utils.message_processor import MessageProcessor


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
    
    工作模式：
    1. 流式模式：实时返回推理过程和生成结果
    2. 非流式模式：等待完整结果后一次性返回
    """

    def __init__(
        self,
        deepseek_api_key: str,
        claude_api_key: str,
        deepseek_api_url: str = None,
        claude_api_url: str = "https://api.anthropic.com/v1/messages",
        claude_provider: str = "anthropic",
        is_origin_reasoning: bool = True
    ):
        """初始化 DeepClaude
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            claude_api_key: Claude API密钥
            deepseek_api_url: DeepSeek API地址，可选
            claude_api_url: Claude API地址，默认为Anthropic官方地址
            claude_provider: Claude服务提供商，默认为"anthropic"
            is_origin_reasoning: 是否使用原始推理格式，默认为True
        """
        self.deepseek_client = DeepSeekClient(
            api_key=deepseek_api_key,
            api_url=deepseek_api_url if deepseek_api_url else "https://api.siliconflow.cn/v1/chat/completions"
        )
        self.claude_client = ClaudeClient(
            api_key=claude_api_key,
            api_url=claude_api_url,
            provider=claude_provider
        )
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-ai/DeepSeek-R1",
        claude_model: str = "claude-3-5-sonnet-20241022"
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程
        
        该方法实现了完整的流式处理流程，包括：
        1. 并发调用 DeepSeek 和 Claude API
        2. 实时返回推理过程和生成结果
        3. 使用队列机制协调数据流转
        4. 提供标准格式的输出流
        
        处理流程：
        1. 初始化：创建会话ID和队列
        2. DeepSeek处理：
           - 调用API获取推理流
           - 收集推理内容
           - 推送到输出队列
        3. Claude处理：
           - 等待推理内容
           - 构造输入消息
           - 调用API获取回答
           - 推送到输出队列
        4. 输出处理：
           - 监控任务完成状态
           - 按序返回数据流
        
        Args:
            messages: 初始消息列表，包含对话历史
            model_arg: 模型参数元组[temperature, top_p, presence_penalty, frequency_penalty]
                - temperature: 温度参数，控制输出的随机性
                - top_p: 核采样参数，控制输出的多样性
                - presence_penalty: 存在惩罚，降低重复token的概率
                - frequency_penalty: 频率惩罚，降低高频token的概率
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            
        Yields:
            字节流数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content
                    }
                }]
            }
            
        异常处理：
        1. DeepSeek API调用异常：记录错误并继续Claude处理
        2. Claude API调用异常：记录错误并结束处理
        3. 队列操作异常：确保正确关闭和清理
        """
        # 验证消息格式
        if not messages:
            error_msg = "消息列表为空"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 检查连续消息
        for i in range(1, len(messages)):
            if messages[i].get("role") == messages[i-1].get("role"):
                error_msg = f"检测到连续的{messages[i].get('role')}消息"
                logger.warning(error_msg)
                raise ValueError(error_msg)
            
        # 转换消息格式
        message_processor = MessageProcessor()
        try:
            messages = message_processor.convert_to_deepseek_format(messages)
            logger.debug(f"转换后的消息: {messages}")
        except Exception as e:
            error_msg = f"消息格式转换失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        # 创建队列，用于收集输出数据
        output_queue = asyncio.Queue()  # 存储最终输出的数据流
        claude_queue = asyncio.Queue()  # 用于传递 DeepSeek 推理内容给 Claude

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []  # 存储完整的推理过程

        async def process_deepseek():
            """处理 DeepSeek 流式的异步函数"""
            logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}, 提供商: {self.deepseek_client.provider}")
            try:
                async for content_type, content in self.deepseek_client.stream_chat(
                    messages=messages, 
                    model=deepseek_model, 
                    is_origin_reasoning=self.is_origin_reasoning
                ):
                    if content_type == "reasoning":
                        # 收集推理内容并构造输出响应
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": content,
                                    "content": None  # 确保设置为 None
                                }
                            }]
                        }
                        logger.debug(f"发送推理响应: {response}")
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    elif content_type == "content":
                        # 推理完成，发送结果给 Claude
                        logger.info(f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}")
                        await claude_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {e}", exc_info=True)
                await claude_queue.put("")
            finally:
                # 标记任务完成
                logger.info("DeepSeek 任务处理完成，标记结束")
                await output_queue.put(None)

        async def process_claude():
            """处理 Claude 流的异步函数
            
            主要职责：
            1. 等待并获取 DeepSeek 的推理结果
            2. 构造 Claude 的输入消息
            3. 调用 Claude API 获取回答
            4. 实时推送回答内容到输出队列
            """
            try:
                # 等待 DeepSeek 的推理结果
                logger.info("等待获取 DeepSeek 的推理内容...")
                reasoning = await claude_queue.get()
                logger.debug(f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}")
                
                # 处理推理内容缺失的情况
                if not reasoning:
                    logger.warning("未能获取到有效的推理内容，将使用默认提示继续")
                    reasoning = "获取推理内容失败"
                    
                # 构造 Claude 的输入消息
                claude_messages = messages.copy()
                combined_content = f"""
                Here's my another model's reasoning process:\n{reasoning}\n\n
                Based on this reasoning, provide your response directly to me:"""
                
                # 处理用户消息，将推理结果添加到最后一条用户消息中
                last_message = claude_messages[-1]
                if last_message.get("role", "") == "user":
                    original_content = last_message["content"]
                    fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
                    last_message["content"] = fixed_content
                    
                # 移除系统消息，因为某些 API 提供商可能不支持
                claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]

                logger.info(f"开始处理 Claude 流，使用模型: {claude_model}, 提供商: {self.claude_client.provider}")

                # 调用 Claude API 获取回答
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model_arg=model_arg,
                    model=claude_model,
                ):
                    if content_type == "answer":
                        # 构造输出响应
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }]
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
            except Exception as e:
                logger.error(f"处理 Claude 流时发生错误: {e}")
            # 标记任务完成
            logger.info("Claude 任务处理完成，标记结束")
            await output_queue.put(None)
        
        # 创建并发任务
        deepseek_task = asyncio.create_task(process_deepseek())
        claude_task = asyncio.create_task(process_claude())
        
        # 等待两个任务完成，通过计数判断
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item
        
        # 发送结束标记
        yield b'data: [DONE]\n\n'

    async def chat_completions_without_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022"
    ) -> dict:
        """处理非流式输出过程
        
        该方法实现了完整的非流式处理流程，主要包括：
        1. 获取 DeepSeek 推理内容
        2. 构造 Claude 输入消息
        3. 计算输入输出的 Token 数量
        4. 生成标准的 OpenAI 格式响应
        
        处理流程：
        1. 初始化：生成会话ID和时间戳
        2. DeepSeek处理：
           - 使用流式方式获取推理内容
           - 累积推理文本
           - 处理异常情况
        3. Claude处理：
           - 构造输入消息
           - 添加推理内容
           - 移除不支持的消息类型
        4. Token计算：
           - 计算输入Token数量
           - 统计输出Token数量
        5. 响应处理：
           - 生成OpenAI格式响应
           - 包含完整的使用统计
        
        Args:
            messages: 初始消息列表，包含对话历史
            model_arg: 模型参数元组[temperature, top_p, presence_penalty, frequency_penalty]
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            
        Returns:
            dict: OpenAI 格式的完整响应，包含以下内容：
            - id: 会话唯一标识
            - object: 响应对象类型
            - created: 创建时间戳
            - model: 使用的模型名称
            - choices: 响应内容列表
              - message: 包含role、content和reasoning_content
              - finish_reason: 结束原因
            - usage: Token使用统计
              - prompt_tokens: 输入Token数量
              - completion_tokens: 输出Token数量
              - total_tokens: 总Token数量
        
        异常处理：
        1. DeepSeek推理异常：记录错误并使用默认值
        2. Claude响应异常：记录错误并向上传递异常
        3. Token计算异常：记录警告并继续处理
        """
        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        reasoning_content = []

        # 1. 获取 DeepSeek 的推理内容（使用流式方式）
        try:
            async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model, self.is_origin_reasoning):
                if content_type == "reasoning":
                    # 收集推理内容
                    reasoning_content.append(content)
                elif content_type == "content":
                    # 推理完成，退出循环
                    break
        except Exception as e:
            # 处理异常情况，使用默认值
            logger.error(f"获取 DeepSeek 推理内容时发生错误: {e}")
            reasoning_content = ["获取推理内容失败"]

        # 2. 构造 Claude 的输入消息
        reasoning = "".join(reasoning_content)  # 合并推理内容
        claude_messages = messages.copy()  # 复制原始消息列表

        # 构造包含推理内容的提示文本
        combined_content = f"""
        Here's my another model's reasoning process:\n{reasoning}\n\n
        Based on this reasoning, provide your response directly to me:"""
        
        # 处理最后一条用户消息，添加推理内容
        last_message = claude_messages[-1]
        if last_message.get("role", "") == "user":
            original_content = last_message["content"]
            fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
            last_message["content"] = fixed_content

        # 移除系统消息，确保兼容性
        claude_messages = [message for message in claude_messages if message.get("role", "") != "system"]

        # 计算输入Token数量
        token_content = "\n".join([message.get("content", "") for message in claude_messages])
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = encoding.encode(token_content)
        logger.debug(f"输入 Tokens: {len(input_tokens)}")

        # 调试输出处理后的消息
        logger.debug("claude messages: " + str(claude_messages))

        # 3. 获取 Claude 的非流式响应
        try:
            answer = ""  # 存储完整响应内容
            # 使用流式方式获取响应，但设置stream=False
            async for content_type, content in self.claude_client.stream_chat(
                messages=claude_messages,
                model_arg=model_arg,
                model=claude_model,
                stream=False
            ):
                if content_type == "answer":
                    # 累积响应内容
                    answer += content
                # 计算输出Token数量
                output_tokens = encoding.encode(answer)
                logger.debug(f"输出 Tokens: {len(output_tokens)}")

            # 4. 构造 OpenAI 格式的完整响应
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_time,
                "model": claude_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "reasoning_content": reasoning
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(input_tokens),
                    "completion_tokens": len(output_tokens),
                    "total_tokens": len(input_tokens + output_tokens)
                }
            }
        except Exception as e:
            # 记录错误并向上传递异常
            logger.error(f"获取 Claude 响应时发生错误: {e}")
            raise e