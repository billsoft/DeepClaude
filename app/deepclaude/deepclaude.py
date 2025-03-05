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
import re  # 添加正则表达式模块
from typing import AsyncGenerator, Dict, List, Any, Optional, Tuple
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient, OllamaR1Client
from app.utils.message_processor import MessageProcessor
import aiohttp
import os
from dotenv import load_dotenv
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

    async def chat_completions_with_stream(self, messages: list, tools: list = None, tool_choice = "auto", **kwargs):
        """处理流式请求，支持思考-回答模式和工具调用

        Args:
            messages: 对话消息列表
            tools: 工具列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Yields:
            bytes: 流式响应数据
        """
        # 初始化变量
        chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
        created_time = kwargs.get("created_time", int(time.time()))
        model_name = kwargs.get("model", "deepclaude")
        claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
        deepseek_model = kwargs.get("deepseek_model", "deepseek-reasoner")
        model_arg = tuple(map(float, os.getenv('MODEL_ARG', '1.0,1.0,0.7,0.1').split(',')))
        model = kwargs.get("model", "deepclaude")

        try:
            logger.info("开始流式处理请求...")
            logger.debug(f"输入消息: {messages}")
            
            # 配置直接透传模式
            direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
            
            # 如果启用直接透传且有工具，直接使用Claude处理
            if direct_tool_pass and tools and len(tools) > 0:
                logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
                
                # 记录工具选择策略
                if isinstance(tool_choice, str):
                    logger.info(f"工具选择策略: {tool_choice}")
                elif isinstance(tool_choice, dict):
                    logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
                else:
                    logger.info(f"工具选择策略: {tool_choice}")
                
                # 转换工具格式
                converted_tools = self._validate_and_convert_tools(tools, target_format='claude-3')
                
                if not converted_tools:
                    logger.warning("没有有效的工具可用，将作为普通对话处理")
                    result = {
                        "id": chat_id,
                        "object": "chat.completion",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "没有找到有效的工具定义，将作为普通对话处理。"
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                    return
                
                logger.info(f"直接使用Claude模型: {claude_model}")
                
                # 准备Claude调用参数
                claude_kwargs = {
                    "messages": messages,
                    "model": claude_model,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "tools": converted_tools
                }
                
                # 工具选择策略转换
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        claude_kwargs["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        # Claude不支持none，将使用空工具列表
                        logger.info("检测到'none'工具选择策略，将不使用工具")
                        claude_kwargs.pop("tools")
                elif isinstance(tool_choice, dict):
                    if tool_choice.get("type") == "function" and "function" in tool_choice:
                        # OpenAI格式转为Claude格式
                        func_name = tool_choice["function"].get("name")
                        if func_name:
                            logger.info(f"指定使用工具: {func_name}")
                            claude_kwargs["tool_choice"] = {
                                "type": "tool",
                                "name": func_name
                            }
                    else:
                        # 已是Claude格式或其他格式
                        claude_kwargs["tool_choice"] = tool_choice
                
                try:
                    # 非流式调用Claude API
                    response = await self.claude_client.chat(**claude_kwargs)
                    
                    # 处理工具调用响应
                    if "tool_calls" in response:
                        tool_calls = response["tool_calls"]
                        logger.info(f"Claude返回了 {len(tool_calls)} 个工具调用")
                        
                        # 构造标准的OpenAI格式响应
                        result = {
                            "id": chat_id,
                            "object": "chat.completion",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": tool_calls
                                },
                                "finish_reason": "tool_calls"
                            }],
                            "usage": {
                                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                        return
                    else:
                        # 处理普通回答响应
                        content = response.get("content", "")
                        if not content and "choices" in response and response["choices"]:
                            content = response["choices"][0].get("message", {}).get("content", "")
                            
                        result = {
                            "id": chat_id,
                            "object": "chat.completion",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                            }
                        }
                        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                        return
                except Exception as e:
                    logger.error(f"直接透传模式下API调用失败: {e}", exc_info=True)
                    # 此处选择回退到推理-回答模式，而不是立即返回错误
                    logger.info("将尝试使用推理-回答模式处理请求")
        except Exception as e:
            logger.error(f"处理流式请求时发生错误: {e}", exc_info=True)
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
            return
        
        # 保存对话到数据库（如果启用）
        if self.save_to_db:
            try:
                user_id = None
                if messages and 'content' in messages[-1]:
                    title = messages[-1]['content'][:20] + "..."
                    user_question = messages[-1]['content']
                else:
                    title = None
                    user_question = "未提供问题内容"
                    
                self.current_conversation_id = self.db_ops.create_conversation(
                    user_id=user_id,
                    title=title
                )
                logger.info(f"创建新对话，ID: {self.current_conversation_id}")
                
                self.db_ops.add_conversation_history(
                    conversation_id=self.current_conversation_id,
                    role="user",
                    content=user_question
                )
                logger.info("用户问题已保存到数据库")
            except Exception as db_e:
                logger.error(f"保存对话数据失败: {db_e}")
        
        # 获取原始问题
        original_question = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        
        # 获取推理内容
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
            
            # 尝试使用不同的推理模式
            for reasoning_mode in self.reasoning_modes[1:]:
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
        
        # 工具调用处理
        has_tool_decision = False
        tool_calls = []
        
        if tools and len(tools) > 0:
            try:
                # 搜索增强
                search_hint = ""
                if self.search_enabled and messages and messages[-1]["role"] == "user":
                    search_hint = await self._enhance_with_search(messages[-1]["content"])
                if search_hint:
                    reasoning = f"{search_hint}\n\n{reasoning}"
                
                # 工具决策
                decision_prompt = self._format_tool_decision_prompt(original_question, reasoning, tools)
                logger.debug(f"工具决策提示: {decision_prompt[:200]}...")
                
                tool_decision_response = await self.claude_client.chat(
                    messages=[{"role": "user", "content": decision_prompt}],
                    model=claude_model,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=model_arg[0],
                    top_p=model_arg[1]
                )
                
                # 处理工具调用决策
                if "tool_calls" in tool_decision_response:
                    tool_calls = tool_decision_response.get("tool_calls", [])
                    has_tool_decision = True
                    logger.info(f"Claude决定使用工具: {len(tool_calls)}个工具调用")
                    
                    # 构造工具调用响应
                    response = {
                        "id": chat_id,
                        "object": "chat.completion",
                        "created": created_time,
                        "model": model,
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
                    yield f"data: {json.dumps(response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                    return
            except Exception as tool_e:
                logger.error(f"工具调用流程失败: {tool_e}", exc_info=True)
        
        # 生成普通回答
        if not has_tool_decision:
            # 构建最终提示
            combined_content = f"""请根据用户的问题和思考过程提供最佳回答:

用户问题: {original_question}

思考过程: 
{reasoning}

要求:
1. 直接回答问题，不要说"根据思考过程"之类的引用
2. 以清晰、准确、有帮助的方式回答
3. 如果思考过程中有不确定性，要明确指出
4. 使用适当的语气和结构
5. 不要重复或解释思考过程"""

            # 发送最终提示到Claude
            claude_messages = [{"role": "user", "content": combined_content}]
            logger.info("正在获取 Claude 回答...")
            
            try:
                # 获取完整回答
                full_content = ""
                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model=claude_model,
                    temperature=model_arg[0],
                    top_p=model_arg[1],
                    stream=False
                ):
                    if content_type in ["answer", "content"]:
                        logger.debug(f"获取到 Claude 回答: {content}")
                        full_content += content
                
                # 保存到数据库（如果启用）
                if self.save_to_db and self.current_conversation_id:
                    try:
                        tokens = len(reasoning.split()) + len(full_content.split())
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
                
                # 构造完整响应
                response = {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }]
                }
                
                yield f"data: {json.dumps(response, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                return
            except Exception as e:
                logger.error(f"获取 Claude 回答失败: {e}")
                error_message = await self._handle_api_error(e)
                result = {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"处理请求时出错: {error_message}"
                        },
                        "finish_reason": "error"
                    }]
                }
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                return

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
        """通用搜索增强函数，不包含特定模式匹配和特定工具假设
        
        Args:
            query: 用户查询内容
            
        Returns:
            str: 搜索增强提示，不含具体搜索结果
        """
        if not self.search_enabled:
            logger.info("搜索增强功能未启用")
            return ""
        
        logger.info(f"考虑为查询提供搜索增强: {query}")
        
        # 不再进行硬编码的模式匹配，而是由工具调用机制决定是否使用搜索
        # 返回一个通用提示，提示模型考虑使用搜索工具
        hint = "此问题可能涉及实时信息，可以考虑使用搜索工具获取最新数据。"
        
        logger.info("已为查询添加搜索增强提示")
        return hint
    
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
            # 通用提取工具名称和参数，支持多种格式
            tool_name = "未知工具"
            tool_args = "{}"
            
            # 从不同格式中提取工具名称和参数
            if "function" in tool_call:
                # OpenAI格式
                func = tool_call.get("function", {})
                tool_name = func.get("name", "未知工具")
                tool_args = func.get("arguments", "{}")
            elif "name" in tool_call:
                # Claude或其他格式
                tool_name = tool_call.get("name", "未知工具")
                tool_args = json.dumps(tool_call.get("input", tool_call.get("arguments", {})), ensure_ascii=False)
            
            # 尝试解析参数为更可读的格式
            try:
                args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                args_str = json.dumps(args_dict, ensure_ascii=False, indent=2)
            except:
                args_str = str(tool_args)
                
            # 通用提取工具结果
            result_content = ""
            if isinstance(tool_result, dict):
                # 尝试多种可能的字段名
                result_content = (tool_result.get("content") or 
                                 tool_result.get("result") or 
                                 tool_result.get("output") or 
                                 tool_result.get("response") or
                                 json.dumps(tool_result, ensure_ascii=False))
            else:
                result_content = str(tool_result)
            
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
6. 必要时可以结合多个工具的结果
7. 不要解释推理过程，直接给出基于工具结果的答案
8. 不要提及您正在使用工具，就像这些信息是您本身知道的一样
"""

        logger.info("向Claude发送工具结果提示生成最终回答")
        try:
            # 将工具结果也添加到提示消息中，以便Claude能够充分利用工具返回的信息
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 使用Claude生成最终回答
            response = await self.claude_client.chat(
                messages=messages,
                model=os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9)
            )
            
            # 提取并返回回答内容
            if "content" in response:
                answer = response.get("content", "")
                logger.info(f"生成最终回答成功(Claude格式): {answer[:100]}...")
                return answer
            elif "choices" in response and response["choices"]:
                answer = response["choices"][0].get("message", {}).get("content", "")
                logger.info(f"生成最终回答成功(OpenAI格式): {answer[:100]}...")
                return answer
            else:
                logger.error(f"未找到回答内容，响应结构: {list(response.keys())}")
                return "抱歉，处理工具结果时出现错误，无法生成回答。"
        except Exception as e:
            logger.error(f"处理工具结果失败: {e}", exc_info=True)
            return f"抱歉，处理工具结果时出现错误: {str(e)}"

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
                # 处理OpenAI格式工具
                function = tool["function"]
                name = function.get("name", "未命名工具")
                description = function.get("description", "无描述")
                parameters = function.get("parameters", {})
                required = parameters.get("required", [])
                properties = parameters.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型")
                    param_description = param_info.get("description", "无描述")
                    enum_values = param_info.get("enum", [])
                    
                    enum_desc = ""
                    if enum_values:
                        enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
                        
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
                
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
            # 处理Claude格式的自定义工具（使用custom字段）
            elif "name" in tool and "custom" in tool:
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "无描述")
                custom = tool.get("custom", {})
                
                # 处理input_schema
                if "input_schema" in custom:
                    input_schema = custom["input_schema"]
                    required = input_schema.get("required", [])
                    properties = input_schema.get("properties", {})
                else:
                    # 兼容直接在custom下的属性
                    required = custom.get("required", [])
                    properties = custom.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型")
                    param_description = param_info.get("description", "无描述")
                    enum_values = param_info.get("enum", [])
                    
                    enum_desc = ""
                    if enum_values:
                        enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
                        
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
                
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
            elif "type" in tool and tool["type"] == "custom":
                # 处理Claude格式工具
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "无描述")
                
                # 尝试解析schema
                tool_schema = tool.get("tool_schema", {})
                required = tool_schema.get("required", [])
                properties = tool_schema.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型")
                    param_description = param_info.get("description", "无描述")
                    enum_values = param_info.get("enum", [])
                    
                    enum_desc = ""
                    if enum_values:
                        enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
                        
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
                
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
            elif "name" in tool and "parameters" in tool:
                # 处理简化格式工具
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "无描述")
                parameters = tool.get("parameters", {})
                required = parameters.get("required", [])
                properties = parameters.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型") 
                    param_description = param_info.get("description", "无描述")
                    enum_values = param_info.get("enum", [])
                    
                    enum_desc = ""
                    if enum_values:
                        enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
                        
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
                
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
            # 处理Claude格式的自定义工具（使用input_schema）
            elif "name" in tool and "input_schema" in tool:
                name = tool.get("name", "未命名工具")
                description = tool.get("description", "无描述")
                input_schema = tool.get("input_schema", {})
                required = input_schema.get("required", [])
                properties = input_schema.get("properties", {})
                
                param_desc = ""
                for param_name, param_info in properties.items():
                    is_required = "必填" if param_name in required else "可选"
                    param_type = param_info.get("type", "未知类型")
                    param_description = param_info.get("description", "无描述")
                    enum_values = param_info.get("enum", [])
                    
                    enum_desc = ""
                    if enum_values:
                        enum_desc = f"，可选值: {', '.join([str(v) for v in enum_values])}"
                        
                    param_desc += f"  - {param_name} ({param_type}, {is_required}): {param_description}{enum_desc}\n"
                    
                tools_description += f"{i}. 工具名称: {name}\n   描述: {description}\n   参数:\n{param_desc}\n"
        
        # 构建完整提示
        prompt = f"""作为一个专业的AI助手，请分析用户问题和我的思考过程，决定是否需要使用工具来完成回答。

用户问题: {original_question}

思考过程: 
{reasoning}

可用工具列表:
{tools_description}

分析要点:
1. 如果问题需要实时或最新信息(例如天气、股票、新闻等)，应使用工具获取
2. 如果问题需要检索特定数据或执行计算，应使用工具
3. 如果问题是一般性知识或推理，不需要使用工具
4. 如果思考过程已包含足够信息回答问题，不需要使用工具

请仔细判断是否使用工具，并给出符合以下格式的回复:

如果决定使用工具，请提供一个有效的工具调用JSON:
```json
{
  "name": "工具名称",
  "arguments": {
    "参数1": "值1",
    "参数2": "值2"
  }
}
```

如果不需要使用工具，只需直接回复: "不需要使用工具"

请注意，你的回复必须是一个有效的JSON对象(如果使用工具)或纯文本短语(如果不使用工具)。不要添加任何解释或评论。"""

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
            # 注意：OpenAI最新规范要求tool_calls作为一个完整的数组
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
            logger.error(f"工具调用响应格式化失败: {e}", exc_info=True)
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

    def _validate_and_convert_tools(self, tools: List[Dict], target_format: str = 'claude-3') -> List[Dict]:
        """验证并转换工具列表，支持多种格式转换
        
        Args:
            tools: 原始工具列表
            target_format: 目标格式
            
        Returns:
            List[Dict]: 验证并转换后的工具列表
        """
        if not tools:
            return []
        
        valid_tools = []
        for i, tool in enumerate(tools):
            # 基本格式检查 
            if not isinstance(tool, dict):
                logger.warning(f"工具格式错误: {tool}")
                continue
            
            # 处理已经是Claude格式的工具
            if "type" in tool and tool["type"] in ["custom", "bash_20250124", "text_editor_20250124"]:
                # 确保工具没有custom字段，这是一个常见错误
                if "custom" in tool:
                    logger.warning(f"检测到工具中的custom字段，这不符合Claude API规范，正在移除: {tool.get('name', '未命名工具')}")
                    # 创建工具的副本避免修改原对象
                    fixed_tool = tool.copy()
                    fixed_tool.pop("custom", None)
                    valid_tools.append(fixed_tool)
                else:
                    valid_tools.append(tool)
                logger.info(f"检测到已是Claude格式的工具: {tool.get('name', '未命名工具')}")
                continue
                
            # 处理OpenAI格式的工具
            if "function" in tool:
                if target_format == 'claude-3':
                    function_data = tool["function"]
                    name = function_data.get("name", "未命名工具")
                    description = function_data.get("description", "")
                    parameters = function_data.get("parameters", {})
                    
                    # 创建Claude格式的自定义工具
                    claude_tool = {
                        "type": "custom",
                        "name": name,
                        "description": description,
                        "tool_schema": parameters
                    }
                    logger.info(f"将OpenAI格式工具 '{name}' 转换为Claude custom格式")
                    valid_tools.append(claude_tool)
                else:
                    # 保持OpenAI格式，确保格式正确
                    # 确保有type字段
                    if "type" not in tool:
                        tool = {"type": "function", "function": tool["function"]}
                    valid_tools.append(tool)
                    logger.info(f"保持OpenAI格式工具: {tool['function'].get('name', '未命名工具')}")
                continue
                
            # 处理Dify格式的工具 (可能用name和api_type字段)
            if "name" in tool and "api_type" in tool:
                logger.info(f"检测到Dify格式工具: {tool.get('name', '未命名工具')}")
                if target_format == 'claude-3':
                    # 尝试从Dify格式转换为Claude格式
                    dify_tool = {
                        "type": "custom",
                        "name": tool.get("name", "未命名工具"),
                        "description": tool.get("description", ""),
                        "tool_schema": tool.get("parameters", {})
                    }
                    valid_tools.append(dify_tool)
                    logger.info(f"已将Dify工具 '{tool.get('name', '未命名工具')}' 转换为Claude格式")
                else:
                    # 转换为OpenAI格式
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", "未命名工具"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    }
                    valid_tools.append(openai_tool)
                    logger.info(f"已将Dify工具 '{tool.get('name', '未命名工具')}' 转换为OpenAI格式")
                continue
                
            # 处理简化格式工具 (仅有name和parameters)
            if "name" in tool and "parameters" in tool:
                logger.info(f"检测到简化格式工具: {tool.get('name', '未命名工具')}")
                if target_format == 'claude-3':
                    simple_tool = {
                        "type": "custom",
                        "name": tool.get("name", "未命名工具"),
                        "description": tool.get("description", ""),
                        "tool_schema": tool.get("parameters", {})
                    }
                    valid_tools.append(simple_tool)
                    logger.info(f"已将简化格式工具转为Claude格式: {tool.get('name', '未命名工具')}")
                else:
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", "未命名工具"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    }
                    valid_tools.append(openai_tool)
                    logger.info(f"已将简化格式工具转为OpenAI格式: {tool.get('name', '未命名工具')}")
                continue
                
            # 处理其他可能的变体格式 (尝试提取关键字段)
            if set(["name", "description"]).issubset(set(tool.keys())):
                logger.info(f"检测到可能的变体格式工具: {tool.get('name', '未命名工具')}")
                
                # 尝试从各种可能的字段中提取参数
                parameters = tool.get("parameters", 
                            tool.get("schema", 
                            tool.get("parameter_schema", 
                            tool.get("tool_schema", {}))))
                
                if target_format == 'claude-3':
                    variant_tool = {
                        "type": "custom",
                        "name": tool.get("name", "未命名工具"),
                        "description": tool.get("description", ""),
                        "tool_schema": parameters
                    }
                    valid_tools.append(variant_tool)
                    logger.info(f"已将变体格式工具转为Claude格式: {tool.get('name', '未命名工具')}")
                else:
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", "未命名工具"),
                            "description": tool.get("description", ""),
                            "parameters": parameters
                        }
                    }
                    valid_tools.append(openai_tool)
                    logger.info(f"已将变体格式工具转为OpenAI格式: {tool.get('name', '未命名工具')}")
                continue
                
            logger.warning(f"工具格式无法识别: {json.dumps(tool, ensure_ascii=False)[:100]}...")
        
        # 日志记录转换结果
        logger.info(f"工具验证和转换完成，原有 {len(tools)} 个工具，有效 {len(valid_tools)} 个工具")
        if valid_tools:
            for i, tool in enumerate(valid_tools):
                if "type" in tool and tool["type"] == "custom":
                    logger.debug(f"有效工具[{i}]: {tool.get('name', '未命名工具')} (Claude格式)")
                else:
                    logger.debug(f"有效工具[{i}]: {tool.get('name', tool.get('function', {}).get('name', '未命名工具'))} (OpenAI格式)")
                
        return valid_tools

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
        tool_choice = "auto",
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'),
        **kwargs
    ) -> dict:
        """处理非流式请求，支持工具调用
        
        Args:
            messages: 对话消息列表
            model_arg: 推理模型参数元组 (temperature, top_p, presence_penalty, frequency_penalty)
            tools: 工具列表
            tool_choice: 工具选择策略
            deepseek_model: DeepSeek推理模型
            claude_model: Claude回答模型
            **kwargs: 其他参数
            
        Returns:
            dict: 完整的响应数据
        """
        logger.info("开始处理非流式请求...")
        logger.debug(f"输入消息: {messages}")
        
        # 配置直接透传模式
        direct_tool_pass = os.getenv('CLAUDE_DIRECT_TOOL_PASS', 'true').lower() == 'true'
        
        # 构造基本响应模板
        chat_id = kwargs.get("chat_id", f"chatcmpl-{uuid.uuid4()}")
        created_time = kwargs.get("created_time", int(time.time()))
        model = kwargs.get("model", "deepclaude")
        
        # 如果启用直接透传且有工具，直接使用Claude处理
        if direct_tool_pass and tools and len(tools) > 0:
            logger.info(f"直接透传模式(非流式): 包含 {len(tools)} 个工具")
            
            # 记录工具选择策略
            if isinstance(tool_choice, str):
                logger.info(f"工具选择策略: {tool_choice}")
            elif isinstance(tool_choice, dict):
                logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
            else:
                logger.info(f"工具选择策略: {tool_choice}")
            
            # 转换工具格式
            converted_tools = self._validate_and_convert_tools(tools, target_format='claude-3')
            
            if not converted_tools:
                logger.warning("没有有效的工具可用，将作为普通对话处理")
                return {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "没有找到有效的工具定义，将作为普通对话处理。"
                        },
                        "finish_reason": "stop"
                    }]
                }
            
            logger.info(f"直接使用Claude模型: {claude_model}")
            
            # 准备Claude调用参数
            claude_kwargs = {
                "messages": messages,
                "model": claude_model,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "tools": converted_tools
            }
            
            # 工具选择策略转换
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    claude_kwargs["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    # Claude不支持none，将使用空工具列表
                    logger.info("检测到'none'工具选择策略，将不使用工具")
                    claude_kwargs.pop("tools")
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function" and "function" in tool_choice:
                    # OpenAI格式转为Claude格式
                    func_name = tool_choice["function"].get("name")
                    if func_name:
                        logger.info(f"指定使用工具: {func_name}")
                        claude_kwargs["tool_choice"] = {
                            "type": "tool",
                            "name": func_name
                        }
                else:
                    # 已是Claude格式或其他格式
                    claude_kwargs["tool_choice"] = tool_choice
            
            try:
                # 非流式调用Claude API
                response = await self.claude_client.chat(**claude_kwargs)
                
                # 处理工具调用响应
                if "tool_calls" in response:
                    tool_calls = response["tool_calls"]
                    logger.info(f"Claude返回了 {len(tool_calls)} 个工具调用")
                    
                    # 构造标准的OpenAI格式响应
                    result = {
                        "id": chat_id,
                        "object": "chat.completion",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {
                            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
                    return result
                else:
                    # 处理普通回答响应
                    content = response.get("content", "")
                    if not content and "choices" in response and response["choices"]:
                        content = response["choices"][0].get("message", {}).get("content", "")
                        
                    result = {
                        "id": chat_id,
                        "object": "chat.completion",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                        }
                    }
                    return result
            except Exception as e:
                logger.error(f"直接透传模式下API调用失败: {e}", exc_info=True)
                # 此处选择回退到推理-回答模式，而不是立即返回错误
                logger.info("将尝试使用推理-回答模式处理请求")
        
        # 剩余的代码保持不变...