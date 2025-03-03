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
import uuid
from typing import AsyncGenerator, Dict, List, Any, Optional, Tuple
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, OllamaR1Client
from app.utils.message_processor import MessageProcessor
import aiohttp
import os
from dotenv import load_dotenv
import re
import sys
import logging
import requests
from datetime import datetime

# 数据库相关导入
from app.database.db_operations import DatabaseOperations
from app.database.db_utils import add_reasoning_column_if_not_exists

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
        """初始化DeepClaude服务
        
        Args:
            **kwargs: 关键字参数
                save_to_db: 是否保存到数据库，默认为False
                db_ops: 数据库操作对象，仅在save_to_db为True时使用
                clients: 客户端对象，用于手动指定客户端，用于测试
                enable_enhanced_reasoning: 是否启用增强推理，默认为True
                claude_api_key: Claude API密钥
                claude_api_url: Claude API URL
                claude_provider: Claude提供商
                deepseek_api_key: DeepSeek API密钥
                deepseek_api_url: DeepSeek API URL
                deepseek_provider: DeepSeek提供商
                ollama_api_url: Ollama API URL
                is_origin_reasoning: 是否使用原始推理格式
        """
        logger.info("初始化DeepClaude服务...")
        
        # 保存传入的配置参数，以便在其他方法中使用
        self.claude_api_key = kwargs.get('claude_api_key', os.getenv('CLAUDE_API_KEY', ''))
        self.claude_api_url = kwargs.get('claude_api_url', os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com/v1/messages'))
        self.claude_provider = kwargs.get('claude_provider', os.getenv('CLAUDE_PROVIDER', 'anthropic'))
        self.ollama_api_url = kwargs.get('ollama_api_url', os.getenv('OLLAMA_API_URL', ''))
        self.is_origin_reasoning = kwargs.get('is_origin_reasoning', os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true')
        
        # 推理内容和模式设置
        self.enable_enhanced_reasoning = kwargs.get('enable_enhanced_reasoning', True)
        self.min_reasoning_chars = 100  # 最小推理字符数量
        self.reasoning_modes = ["auto", "chain-of-thought", "zero-shot"]  # 推理模式列表
        self.saved_reasoning = ""  # 保存的推理内容，用于诊断
        self.processor = MessageProcessor()  # 消息处理器
        
        # 定义推理提供者
        self.reasoning_providers = {
            'deepseek': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
                api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
                provider=os.getenv('DEEPSEEK_PROVIDER', 'deepseek')
            ),
            'siliconflow': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
                api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
                provider='siliconflow'
            ),
            'nvidia': lambda: DeepSeekClient(
                api_key=kwargs.get('deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', '')),
                api_url=kwargs.get('deepseek_api_url', os.getenv('DEEPSEEK_API_URL', '')),
                provider='nvidia'
            ),
            'ollama': lambda: OllamaR1Client(
                api_url=kwargs.get('ollama_api_url', os.getenv('OLLAMA_API_URL', ''))
            )
        }
        
        # 支持的工具列表，不实际执行这些工具，只返回标准格式的响应
        self.supported_tools = {
            "search": {
                "name": "search",
                "description": "搜索网络获取实时信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询内容"
                        }
                    },
                    "required": ["query"]
                }
            },
            "weather": {
                "name": "weather",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "地点名称，如：北京、上海"
                        },
                        "date": {
                            "type": "string",
                            "description": "日期，如：today、tomorrow",
                            "enum": ["today", "tomorrow"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        # 添加工具格式转换配置
        self.tool_format_mapping = {
            'gpt-4': {
                'type': 'function',
                'function_field': 'function'
            },
            'gpt-3.5-turbo': {
                'type': 'function',
                'function_field': 'function'
            },
            'claude-3': {
                'type': 'tool',
                'function_field': 'function'
            }
        }
        
        # 数据库相关设置
        self.save_to_db = kwargs.get('save_to_db', os.getenv('SAVE_TO_DB', 'false').lower() == 'true')
        if self.save_to_db:
            logger.info("启用数据库存储...")
            self.db_ops = kwargs.get('db_ops', DatabaseOperations())
            self.current_conversation_id = None
            # 检查并添加reasoning列（如果不存在）
            add_reasoning_column_if_not_exists()
        else:
            logger.info("数据库存储已禁用")
            self.db_ops = None
            
        # 初始化客户端
        if 'clients' in kwargs:
            self.thinker_client = kwargs['clients'].get('thinker')
            self.claude_client = kwargs['clients'].get('claude')
        else:
            logger.info("初始化思考者客户端...")
            provider = self._get_reasoning_provider()
            self.thinker_client = provider
            
            logger.info("初始化Claude客户端...")
            self.claude_client = ClaudeClient(
                api_key=self.claude_api_key,
                api_url=self.claude_api_url,
                provider=self.claude_provider
            )
        
        # 验证配置有效性
        self._validate_config()
        
        # 配置搜索增强
        self.search_enabled = os.getenv('ENABLE_SEARCH_ENHANCEMENT', 'true').lower() == 'true'
        
        logger.info("DeepClaude服务初始化完成")
        
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

    async def chat_completions_with_stream(self, messages: list, tools: list = None, tool_choice: str = "auto", **kwargs):
        """流式对话完成，支持工具调用"""
        try:
            logger.info("开始流式处理请求...")
            logger.debug(f"输入消息: {messages}")
            
            # 1. 准备参数和变量
            chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
            created_time = kwargs.get("created_time", int(time.time()))
            model_name = kwargs.get("model", "deepclaude")
            has_tools = tools and len(tools) > 0
            
            # 验证工具配置
            if has_tools:
                logger.info(f"请求包含 {len(tools)} 个工具")
                logger.info("原始工具格式:")
                logger.info(json.dumps(tools, ensure_ascii=False))
                
                # 验证并转换工具
                tools = self._validate_and_convert_tools(tools, target_format='claude-3')
                if tools:
                    logger.info(f"验证成功 {len(tools)} 个工具")
                    logger.info("转换后的工具格式:")
                    logger.info(json.dumps(tools, ensure_ascii=False))
                else:
                    logger.warning("没有有效的工具可用，将作为普通对话处理")
                    has_tools = False
                
                logger.info(f"工具选择策略: {tool_choice}")
            else:
                logger.info("请求中不包含工具，将作为普通对话处理")
            
            # 提取原始问题
            original_question = ""
            if messages and messages[-1]["role"] == "user":
                original_question = messages[-1]["content"]
                logger.info(f"原始问题: {original_question}")
                
                # 分析问题是否需要工具
                if has_tools:
                    logger.info("分析问题是否需要工具...")
                    need_weather = any(word in original_question.lower() for word in ["天气", "气温", "weather"])
                    need_search = any(word in original_question.lower() for word in ["搜索", "查询", "search"])
                    
                    if need_weather:
                        logger.info("检测到天气查询需求")
                    if need_search:
                        logger.info("检测到搜索查询需求")
                    
                    if not (need_weather or need_search):
                        logger.info("未检测到明确的工具需求")
            
            # 2. 思考阶段
            logger.info("开始思考阶段...")
            search_enhanced = False
            search_hint = ""
            
            if has_tools and self.search_enabled and original_question:
                search_hint = await self._enhance_with_search(original_question)
                if search_hint:
                    search_enhanced = True
                    logger.info("使用搜索增强思考")
                    yield self._format_stream_response(
                        f"使用搜索增强思考...\n{search_hint}",
                        content_type="reasoning",
                        is_first_thought=True,
                        **kwargs
                    )
            
            if not search_enhanced:
                yield self._format_stream_response(
                    "开始思考问题...",
                    content_type="reasoning",
                    is_first_thought=True,
                    **kwargs
                )
            
            # 获取推理内容
            reasoning_content = []
            reasoning_success = False
            thought_complete = False
            full_reasoning = ""
            
            try:
                provider = self._get_reasoning_provider()
                logger.info(f"使用推理提供者: {provider.__class__.__name__}")
                
                for retry_count, reasoning_mode in enumerate(self.reasoning_modes):
                    if reasoning_success:
                        break
                        
                    if retry_count > 0:
                        logger.info(f"尝试使用不同的推理模式: {reasoning_mode}")
                        os.environ["DEEPSEEK_REASONING_MODE"] = reasoning_mode
                        provider = self._get_reasoning_provider()
                    
                    try:
                        async for content_type, content in provider.get_reasoning(
                            messages=messages,
                            **self._prepare_thinker_kwargs(kwargs)
                        ):
                            if content_type == "reasoning":
                                reasoning_content.append(content)
                                logger.debug(f"收到推理内容: {content[:50]}...")
                                yield self._format_stream_response(
                                    content,
                                    content_type="reasoning",
                                    is_first_thought=False,
                                    **kwargs
                                )
                            elif content_type == "content":
                                thought_complete = True
                                logger.debug("推理阶段完成")
                        
                        if len("".join(reasoning_content)) > self.min_reasoning_chars:
                            reasoning_success = True
                            logger.info("成功获取足够的推理内容")
                    except Exception as e:
                        logger.error(f"推理获取失败 (模式: {reasoning_mode}): {e}")
                        
                        if retry_count == len(self.reasoning_modes) - 1:
                            error_message = await self._handle_api_error(e)
                            logger.error(f"所有推理模式都失败: {error_message}")
                            yield self._format_stream_response(
                                f"思考过程中遇到错误: {error_message}",
                                content_type="error",
                                is_first_thought=False,
                                **kwargs
                            )
                
                full_reasoning = "\n".join(reasoning_content)
                logger.info(f"推理内容长度: {len(full_reasoning)} 字符")
                
                if search_hint:
                    full_reasoning = f"{search_hint}\n\n{full_reasoning}"
                    logger.debug("已添加搜索提示到推理内容")
                
                # 3. 工具调用阶段
                if has_tools and reasoning_success:
                    logger.info(f"开始工具调用决策 - 工具数量: {len(tools)}")
                    
                    # 决定是否需要使用工具
                    decision_prompt = self._format_tool_decision_prompt(
                        original_question=original_question,
                        reasoning=full_reasoning,
                        tools=tools
                    )
                    logger.debug(f"工具决策提示: {decision_prompt[:200]}...")
                    
                    # 向Claude发送决策请求
                    tool_decision_response = await self.claude_client.chat(
                        messages=[{"role": "user", "content": decision_prompt}],
                        tools=tools,
                        tool_choice=tool_choice,
                        **self._prepare_answerer_kwargs(kwargs)
                    )
                    
                    # 如果决定使用工具，返回工具调用响应
                    if "tool_calls" in tool_decision_response.get("choices", [{}])[0].get("message", {}):
                        tool_calls = tool_decision_response["choices"][0]["message"]["tool_calls"]
                        if tool_calls:
                            tool_names = [t.get("function", {}).get("name", "未知工具") for t in tool_calls]
                            logger.info(f"工具调用决策结果: 使用工具 {', '.join(tool_names)}")
                            
                            for tool_call in tool_calls:
                                logger.info(f"生成工具调用响应: {tool_call.get('function', {}).get('name', '未知工具')}")
                                yield self._format_tool_call_response(
                                    tool_call=tool_call,
                                    chat_id=chat_id,
                                    created_time=created_time,
                                    model=model_name
                                )
                            
                            logger.info("工具调用流程结束，等待客户端执行工具")
                            yield b'data: [DONE]\n\n'
                            return
                        else:
                            logger.info("工具调用决策结果: 不使用工具")
                    else:
                        logger.info("工具调用决策结果: 不使用工具")
                
                # 4. 回答阶段
                logger.info("开始生成最终回答...")
                yield self._format_stream_response(
                    "\n\n---\n思考完毕，开始回答：\n\n",
                    content_type="separator",
                    is_first_thought=False,
                    **kwargs
                )
                
                # 构造Claude的输入消息
                combined_content = f"""
这是我自己基于问题的思考过程:\n{full_reasoning}\n\n
上面是我自己的思考过程不一定完全正确请借鉴思考过程和期中你也认为正确的部分（1000% 权重）
，现在请给出详细和细致的答案，不要省略步骤和步骤细节
，要分解原题确保你理解了原题的每个部分，也要掌握整体意思
，最佳质量（1000% 权重），最详细解答（1000% 权重），不要回答太简单让我能参考一步步应用（1000% 权重）:"""
                
                claude_messages = [{"role": "user", "content": combined_content}]
                logger.debug("向Claude发送最终提示")
                
                # 流式获取Claude回答
                answer_content = []
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    **self._prepare_answerer_kwargs(kwargs)
                ):
                    if content_type in ["answer", "content"]:
                        answer_content.append(content)
                        logger.debug(f"收到回答内容: {content[:50]}...")
                        yield self._format_stream_response(
                            content,
                            content_type="content",
                            is_first_thought=False,
                            **kwargs
                        )
                
                logger.info("回答生成完成")
                
                # 发送流式响应结束标志
                yield b'data: [DONE]\n\n'
                
            except Exception as e:
                logger.error(f"流式处理过程中出错: {e}", exc_info=True)
                error_message = await self._handle_api_error(e)
                yield self._format_stream_response(
                    f"处理请求时出错: {error_message}",
                    content_type="error",
                    is_first_thought=False,
                    **kwargs
                )
                
        except Exception as outer_e:
            logger.error(f"流式处理外层错误: {outer_e}", exc_info=True)
            yield self._format_stream_response(
                f"服务器错误: {str(outer_e)}",
                content_type="error",
                is_first_thought=False,
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
            if provider_type == 'deepseek':
                model = 'deepseek-reasoner'  # 使用确定可用的模型
            elif provider_type == 'siliconflow':
                model = 'deepseek-ai/DeepSeek-R1'
            elif provider_type == 'nvidia':
                model = 'deepseek-ai/deepseek-r1'
            
        return {
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9)
        }
        
    def _prepare_answerer_kwargs(self, kwargs: dict) -> dict:
        """准备回答者参数"""
        return {
            'model': os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
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
        tools: list = None,
        tool_choice: str = "auto",
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
        **kwargs
    ) -> dict:
        """非流式对话完成
        
        Args:
            messages: 对话消息列表
            model_arg: 模型参数元组
            tools: 工具列表
            tool_choice: 工具选择策略
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            
        Returns:
            dict: 包含回答内容的响应字典
        """
        logger.info("开始处理请求...")
        logger.debug(f"输入消息: {messages}")
        
        # 创建或获取对话ID
        if self.save_to_db:
            try:
                # 提取用户ID（如果有的话）
                user_id = None  # 非流式模式通常不传递用户ID，使用默认管理员用户
                
                # 从最后一条消息中提取标题（取前20个字符作为对话标题）
                if messages and 'content' in messages[-1]:
                    title = messages[-1]['content'][:20] + "..."
                    # 保存用户问题
                    user_question = messages[-1]['content']
                else:
                    title = None
                    user_question = "未提供问题内容"
                    
                # 创建新对话
                self.current_conversation_id = self.db_ops.create_conversation(
                    user_id=user_id, 
                    title=title
                )
                logger.info(f"创建新对话，ID: {self.current_conversation_id}")
                
                # 保存用户问题
                self.db_ops.add_conversation_history(
                    conversation_id=self.current_conversation_id,
                    role="user",
                    content=user_question
                )
                logger.info("用户问题已保存到数据库")
            except Exception as db_e:
                logger.error(f"保存对话数据失败: {db_e}")
        
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
        
        original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        has_tool_decision = False
        tool_calls = []
        
        # 如果提供了工具参数，确定是否需要使用工具
        if tools and len(tools) > 0:
            try:
                # 如果启用了搜索增强，可以添加提示
                search_hint = ""
                if self.search_enabled and messages and messages[-1]["role"] == "user":
                    search_hint = await self._enhance_with_search(messages[-1]["content"])
                    if search_hint:
                        reasoning = f"{search_hint}\n\n{reasoning}"
                
                # 针对工具的决策提示
                decision_prompt = self._format_tool_decision_prompt(original_question, reasoning, tools)
                logger.debug(f"工具决策提示: {decision_prompt[:200]}...")
                
                # 向Claude发送决策请求
                tool_decision_response = await self.claude_client.chat(
                    messages=[{"role": "user", "content": decision_prompt}],
                    model=claude_model,
                    tools=tools,
                    tool_choice=tool_choice,
                    model_arg=model_arg
                )
                
                # 如果Claude决定使用工具，返回工具调用响应
                if "tool_calls" in tool_decision_response:
                    tool_calls = tool_decision_response.get("tool_calls", [])
                    has_tool_decision = True
                    logger.info(f"Claude决定使用工具: {len(tool_calls)}个工具调用")
                    
                    # 构造OpenAI格式的响应
                    response = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": kwargs.get("model", "deepclaude"),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }
                    
                    return response
            except Exception as tool_e:
                logger.error(f"工具调用流程失败: {tool_e}")
                # 如果工具调用失败，回退到普通回答
        
        # 如果没有工具调用或决策不使用工具，生成普通回答
        if not has_tool_decision:
            # 构造 Claude 的输入消息
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
                
                # 保存AI回答到数据库
                if self.save_to_db and self.current_conversation_id:
                    try:
                        # 估算token数量
                        tokens = len(reasoning.split()) + len(full_content.split())
                        
                        # 保存AI回答
                        self.db_ops.add_conversation_history(
                            conversation_id=self.current_conversation_id,
                            role="ai",
                            content=full_content,
                            reasoning=reasoning,
                            model_name=claude_model,
                            tokens=tokens
                        )
                        logger.info("AI回答和思考过程已保存到数据库")
                    except Exception as db_e:
                        logger.error(f"保存AI回答数据失败: {db_e}")
                
                # 返回OpenAI格式的响应
                response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": kwargs.get("model", "deepclaude"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }]
                }
                
                return response
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
        
        # 检查provider是否在支持列表中
        if provider not in self.reasoning_providers:
            raise ValueError(f"不支持的推理提供者: {provider}")
            
        # 针对不同提供者进行验证
        if provider == 'deepseek':
            if not os.getenv('DEEPSEEK_API_KEY'):
                raise ValueError("使用 DeepSeek 时必须提供 API KEY")
            if not os.getenv('DEEPSEEK_API_URL'):
                raise ValueError("使用 DeepSeek 时必须提供 API URL")
        elif provider == 'ollama':
            if not os.getenv('OLLAMA_API_URL'):
                raise ValueError("使用 Ollama 时必须提供 API URL")
        elif provider == 'siliconflow':
            if not os.getenv('DEEPSEEK_API_KEY'):
                raise ValueError("使用 硅基流动 时必须提供 DeepSeek API KEY")
            if not os.getenv('DEEPSEEK_API_URL'):
                raise ValueError("使用 硅基流动 时必须提供 DeepSeek API URL")
        elif provider == 'nvidia':
            if not os.getenv('DEEPSEEK_API_KEY'):
                raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API KEY")
            if not os.getenv('DEEPSEEK_API_URL'):
                raise ValueError("使用 NVIDIA 时必须提供 DeepSeek API URL")
                
        # 验证Claude配置
        if not os.getenv('CLAUDE_API_KEY'):
            raise ValueError("必须提供 CLAUDE_API_KEY 环境变量")

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

    # 搜索增强函数
    async def _enhance_with_search(self, query: str) -> str:
        """使用搜索增强查询，获取最新信息
        注意：实际搜索操作应由Dify执行，这里仅返回提示文本
        
        Args:
            query: 用户查询内容
            
        Returns:
            str: 搜索结果提示文本
        """
        if not self.search_enabled:
            logger.info("搜索增强功能未启用")
            return ""
        
        logger.info(f"建议使用搜索增强查询: {query}")
        return "建议使用搜索工具获取最新信息。"
    
    async def _handle_tool_results(self, original_question: str, reasoning: str, 
                                   tool_calls: List[Dict], tool_results: List[Dict], **kwargs) -> str:
        """处理工具调用结果并生成最终回答
        
        Args:
            original_question: 原始用户问题
            reasoning: 推理内容
            tool_calls: 工具调用列表
            tool_results: 工具调用结果列表
            **kwargs: 其他参数
            
        Returns:
            str: 最终回答
        """
        logger.info(f"处理工具调用结果 - 工具数: {len(tool_calls)}, 结果数: {len(tool_results)}")
        
        # 构建工具调用及结果的详细描述
        tools_info = ""
        for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
            func = tool_call.get("function", {})
            tool_name = func.get("name", "未知工具")
            tool_args = func.get("arguments", "{}")
            
            # 尝试解析参数为更可读的格式
            try:
                args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                args_str = json.dumps(args_dict, ensure_ascii=False, indent=2)
            except:
                args_str = str(tool_args)
                
            # 处理结果
            result_content = tool_result.get("content", "")
            
            tools_info += f"""
工具 {i}: {tool_name}
参数:
{args_str}

结果:
{result_content}
"""

        # 构建完整提示
        prompt = f"""请根据以下信息生成一个完整、准确的回答：

用户问题: {original_question}

思考过程:
{reasoning}

工具调用结果:
{tools_info}

要求：
1. 直接使用工具返回的数据回答问题
2. 确保回答完全解决用户的问题
3. 使用清晰、易懂的语言
4. 如果工具结果不完整或有错误，要说明情况
5. 回答要有逻辑性和连贯性
6. 必要时可以结合多个工具的结果"""
        
        logger.info("向Claude发送工具结果提示生成最终回答")
        
        # 向Claude发送请求生成最终回答
        response = await self.claude_client.chat(
            messages=[{"role": "user", "content": prompt}],
            **self._prepare_answerer_kwargs(kwargs)
        )
        
        if "choices" in response and response["choices"]:
            answer = response["choices"][0]["message"]["content"]
            logger.info(f"生成最终回答成功: {answer[:100]}...")
            return answer
        else:
            logger.warning("生成最终回答失败")
            return "抱歉，无法处理工具调用结果。"

    def _format_tool_decision_prompt(self, original_question: str, reasoning: str, tools: List[Dict]) -> str:
        """格式化工具决策提示，用于Claude判断是否需要使用工具
        
        Args:
            original_question: 原始用户问题
            reasoning: 推理内容
            tools: 可用工具列表
            
        Returns:
            str: 格式化的提示文本
        """
        # 提取工具描述
        tools_description = ""
        for i, tool in enumerate(tools, 1):
            if "function" in tool:
                function = tool["function"]
                name = function.get("name", "未命名工具")
                description = function.get("description", "无描述")
                
                # 提取参数信息
                parameters = function.get("parameters", {})
                required = parameters.get("required", [])
                properties = parameters.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型")
                    param_description = param_info.get("description", "无描述")
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}\n"
                
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
        
        # 构造决策提示
        prompt = f"""你是一个工具调用决策专家。请分析用户问题和推理内容，判断是否需要使用工具获取额外信息。

用户问题: {original_question}

推理内容:
{reasoning}

可用工具:
{tools_description}

判断标准:
1. 如果问题需要实时数据（如天气、搜索等），必须使用相应工具
2. 如果问题是关于常识或可以通过推理解决，则不需要工具
3. 如果推理内容已提供足够信息，则不需要工具

如果需要使用工具，请直接返回工具调用请求，格式如下：
{{
  "tool_calls": [
    {{
      "id": "call_xxxxx",  // 8位唯一ID
      "type": "function",
      "function": {{
        "name": "工具名称",
        "arguments": {{
          // 具体参数
        }}
      }}
    }}
  ]
}}

注意事项:
1. 只在确实需要额外信息时才使用工具
2. 参数值必须是具体的值，不要使用占位符
3. 必须提供所有必填参数
4. 参数值要符合实际场景，如城市名、日期等
5. 工具调用ID必须是唯一的8位字符串
6. 返回的必须是有效的JSON格式"""
        
        logger.info(f"生成工具决策提示 - 问题: '{original_question[:30]}...'")
        return prompt

    def _format_tool_call_response(self, tool_call: Dict, **kwargs) -> bytes:
        """格式化工具调用响应，确保完全符合OpenAI API规范
        
        Args:
            tool_call: 工具调用信息
            **kwargs: 其他参数
            
        Returns:
            bytes: 格式化的SSE响应
        """
        try:
            # 确保工具调用ID存在且格式正确
            tool_call_id = tool_call.get("id")
            if not tool_call_id or not isinstance(tool_call_id, str) or len(tool_call_id) < 8:
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            # 确保函数名和参数格式正确
            function = tool_call.get("function", {})
            function_name = function.get("name", "")
            function_args = function.get("arguments", "{}")
            
            # 如果参数不是字符串格式，转换为正确的JSON字符串
            if not isinstance(function_args, str):
                try:
                    function_args = json.dumps(function_args, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"参数序列化失败: {e}")
                    function_args = "{}"
            
            # 构造标准的OpenAI API响应格式
            response = {
                "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
                "object": "chat.completion.chunk",
                "created": kwargs.get("created_time", int(time.time())),
                "model": kwargs.get("model", "deepclaude"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": function_args
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
            
            logger.info(f"工具调用响应格式化完成 - 工具: {function_name}, ID: {tool_call_id}")
            return f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode('utf-8')
            
        except Exception as e:
            logger.error(f"工具调用响应格式化失败: {e}")
            # 返回一个错误响应
            error_response = {
                "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
                "object": "chat.completion.chunk",
                "created": kwargs.get("created_time", int(time.time())),
                "model": kwargs.get("model", "deepclaude"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"工具调用失败: {str(e)}"
                    },
                    "finish_reason": "error"
                }]
            }
            return f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')
    
    def _format_tool_result_response(self, tool_result: Dict, **kwargs) -> bytes:
        """格式化工具结果响应，用于流式输出
        
        Args:
            tool_result: 工具执行结果
            **kwargs: 其他参数
            
        Returns:
            bytes: 格式化的SSE响应
        """
        try:
            # 验证工具结果格式
            if not isinstance(tool_result, dict):
                raise ValueError("工具结果必须是字典格式")
            
            tool_call_id = tool_result.get("tool_call_id")
            if not tool_call_id:
                raise ValueError("工具结果必须包含tool_call_id")
            
            content = tool_result.get("content")
            if content is None:
                raise ValueError("工具结果必须包含content")
            
            # 构造标准的OpenAI API响应格式
            response = {
                "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
                "object": "chat.completion.chunk",
                "created": kwargs.get("created_time", int(time.time())),
                "model": kwargs.get("model", "deepclaude"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id
                    },
                    "finish_reason": None
                }]
            }
            
            logger.info(f"工具结果响应格式化完成 - ID: {tool_call_id}")
            return f"data: {json.dumps(response, ensure_ascii=False)}\n\n".encode('utf-8')
            
        except Exception as e:
            logger.error(f"工具结果响应格式化失败: {e}")
            # 返回一个错误响应
            error_response = {
                "id": kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}"),
                "object": "chat.completion.chunk",
                "created": kwargs.get("created_time", int(time.time())),
                "model": kwargs.get("model", "deepclaude"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"工具结果处理失败: {str(e)}"
                    },
                    "finish_reason": "error"
                }]
            }
            return f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')

    def _validate_tool(self, tool: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """验证工具格式并尝试修复
        
        Args:
            tool: 工具配置字典
            
        Returns:
            Tuple[bool, str, Optional[Dict]]: 
                - 是否有效
                - 错误信息
                - 修复后的工具配置（如果可以修复）
        """
        if not isinstance(tool, dict):
            return False, f"工具必须是字典格式，当前类型: {type(tool)}", None
            
        # 检查必要字段
        if "function" not in tool and "type" not in tool:
            return False, "工具缺少必要字段 'function' 或 'type'", None
            
        # 如果是 type 格式，尝试转换为 function 格式
        if "type" in tool and tool["type"] in ["function", "tool"]:
            function_field = "function" if tool["type"] == "function" else "parameters"
            if function_field in tool:
                tool = {
                    "function": tool[function_field]
                }
                
        # 验证 function 字段
        function = tool.get("function", {})
        if not isinstance(function, dict):
            return False, f"function 必须是字典格式，当前类型: {type(function)}", None
            
        # 检查必要的 function 字段
        if "name" not in function:
            return False, "function 缺少必要字段 'name'", None
            
        # 验证参数格式
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            return False, f"parameters 必须是字典格式，当前类型: {type(parameters)}", None
            
        # 尝试修复常见问题
        fixed_tool = {
            "function": {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": {
                    "type": parameters.get("type", "object"),
                    "properties": parameters.get("properties", {}),
                    "required": parameters.get("required", [])
                }
            }
        }
        
        return True, "", fixed_tool
        
    def _validate_and_convert_tools(self, tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
        """验证并转换工具列表
        
        Args:
            tools: 原始工具列表
            target_format: 目标格式
            
        Returns:
            List[Dict]: 验证并转换后的工具列表
        """
        if not tools:
            return []
            
        valid_tools = []
        for tool in tools:
            # 验证工具格式
            is_valid, error_msg, fixed_tool = self._validate_tool(tool)
            if not is_valid:
                logger.warning(f"工具验证失败: {error_msg}")
                continue
                
            tool_to_use = fixed_tool or tool
            
            # 检查工具是否在支持列表中
            func = tool_to_use.get("function", {})
            if func.get("name") in self.supported_tools:
                # 转换为目标格式
                format_config = self.tool_format_mapping.get(target_format)
                if format_config:
                    converted_tool = {
                        'type': format_config['type'],
                        format_config['function_field']: func
                    }
                    valid_tools.append(converted_tool)
                    logger.info(f"工具 {func.get('name')} 验证成功并转换为 {target_format} 格式")
                else:
                    valid_tools.append(tool_to_use)
                    logger.info(f"工具 {func.get('name')} 验证成功，保持原始格式")
            else:
                logger.warning(f"工具 {func.get('name')} 不在支持列表中")
                
        return valid_tools