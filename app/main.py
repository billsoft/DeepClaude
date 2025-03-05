# 导入系统相关模块
import os  # 用于操作系统相关功能，如环境变量读取
import sys  # 用于系统相关功能，如程序退出
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量
import uuid
import time
import json

# 加载环境变量配置
# 从.env文件中读取配置，支持本地开发环境和生产环境的配置分离
load_dotenv()

# 设置代理配置
def setup_proxy():
    """设置代理配置
    根据环境变量ENABLE_PROXY的值来设置或清除系统代理
    """
    enable_proxy = os.getenv('ENABLE_PROXY', 'false').lower() == 'true'
    if enable_proxy:
        # 设置代理
        os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY', '')
        os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY', '')
    else:
        # 清除代理
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

# 在导入FastAPI之前设置代理
setup_proxy()

# 导入FastAPI相关依赖
from fastapi import FastAPI, Depends, Request  # FastAPI框架核心组件
from fastapi.responses import StreamingResponse  # 用于处理流式响应
from fastapi.middleware.cors import CORSMiddleware  # 处理跨域请求的中间件
from app.utils.logger import logger  # 日志工具
from app.utils.auth import verify_api_key  # API密钥验证
from fastapi.responses import JSONResponse  # 用于返回JSON响应

# 导入API路由
from app.api.v1.deepclaude import router as deepclaude_router
# 导入DeepClaude类
from app.deepclaude import DeepClaude

app = FastAPI(title="DeepClaude API")

"""DeepClaude API 服务主模块

本模块实现了DeepClaude的FastAPI服务器，提供以下主要功能：
1. 环境变量配置管理
2. FastAPI应用初始化和CORS中间件配置
3. API路由定义（支持流式/非流式输出）
4. 请求参数验证和错误处理
5. 与Claude和DeepSeek模型的交互封装

主要API端点：
- GET /: 服务健康检查
- GET /v1/models: 获取支持的模型列表
- POST /v1/chat/completions: 处理聊天补全请求
- POST /test_tool_call: 测试工具调用功能

环境变量配置：
- ALLOW_ORIGINS: CORS允许的源，多个源用逗号分隔
- CLAUDE_API_KEY: Claude API密钥
- CLAUDE_MODEL: 使用的Claude模型名称
- CLAUDE_PROVIDER: Claude服务提供商(anthropic/openrouter/oneapi)
- CLAUDE_API_URL: Claude API地址
- DEEPSEEK_API_KEY: DeepSeek API密钥
- DEEPSEEK_API_URL: DeepSeek API地址
- DEEPSEEK_MODEL: 使用的DeepSeek模型名称
- IS_ORIGIN_REASONING: 是否使用原始推理格式
"""

# 从环境变量获取 CORS配置, API 密钥、地址以及模型名称
# ALLOW_ORIGINS: 允许跨域请求的源，可以是单个域名或用逗号分隔的多个域名，默认为"*"表示允许所有源
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")

# Claude相关配置
# CLAUDE_API_KEY: Claude API访问密钥
# CLAUDE_MODEL: 使用的Claude模型版本
# CLAUDE_PROVIDER: Claude服务提供商，支持anthropic/openrouter/oneapi
# CLAUDE_API_URL: Claude API的访问地址
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER", "anthropic") # Claude模型提供商, 默认为anthropic
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")

# DeepSeek相关配置
# DEEPSEEK_API_KEY: DeepSeek API访问密钥
# DEEPSEEK_API_URL: DeepSeek API的访问地址
# DEEPSEEK_MODEL: 使用的DeepSeek模型版本
# DEEPSEEK_PROVIDER: DeepSeek 提供商, 默认为 deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
DEEPSEEK_PROVIDER = os.getenv("DEEPSEEK_PROVIDER", "deepseek")

# 是否使用原始推理格式
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "false").lower() == "true"

# 获取思考者提供商配置
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "deepseek").lower()

# 为硅基流动和NVIDIA提供商设置必要的推理模式
if REASONING_PROVIDER in ['siliconflow', 'nvidia']:
    if os.getenv('DEEPSEEK_REASONING_MODE', 'auto') != 'reasoning_field':
        logger.warning(f"硅基流动/NVIDIA提供商推荐使用reasoning_field推理模式，当前模式为: {os.getenv('DEEPSEEK_REASONING_MODE', 'auto')}")
        logger.warning("已自动设置为reasoning_field模式")
        os.environ['DEEPSEEK_REASONING_MODE'] = 'reasoning_field'
    
    if not os.getenv('IS_ORIGIN_REASONING', 'false').lower() == 'true':
        logger.warning("硅基流动/NVIDIA提供商需要启用原始推理格式，已自动设置IS_ORIGIN_REASONING=true")
        os.environ['IS_ORIGIN_REASONING'] = 'true'

# CORS中间件配置
# allow_origins_list: 允许的源列表，从ALLOW_ORIGINS环境变量解析
# allow_credentials: 允许携带认证信息
# allow_methods: 允许的HTTP方法
# allow_headers: 允许的HTTP头部
allow_origins_list = ALLOW_ORIGINS.split(",") if ALLOW_ORIGINS else [] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加 Ollama 配置获取
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
if REASONING_PROVIDER == 'ollama' and not OLLAMA_API_URL:
    logger.critical("使用 Ollama 推理时必须设置 OLLAMA_API_URL")
    sys.exit(1)

# 验证必要的配置
if REASONING_PROVIDER == 'deepseek' and not DEEPSEEK_API_KEY:
    logger.critical("使用 DeepSeek 推理时必须设置 DEEPSEEK_API_KEY")
    sys.exit(1)

# 验证硅基流动配置
if REASONING_PROVIDER == 'siliconflow' and not DEEPSEEK_API_KEY:
    logger.critical("使用硅基流动推理时必须设置 DEEPSEEK_API_KEY")
    sys.exit(1)

# 验证NVIDIA配置
if REASONING_PROVIDER == 'nvidia' and not DEEPSEEK_API_KEY:
    logger.critical("使用NVIDIA推理时必须设置 DEEPSEEK_API_KEY")
    sys.exit(1)

if not CLAUDE_API_KEY:
    logger.critical("必须设置 CLAUDE_API_KEY")
    sys.exit(1)

# 初始化DeepClaude实例
deep_claude = DeepClaude(
    claude_api_key=CLAUDE_API_KEY,
    claude_api_url=CLAUDE_API_URL,
    claude_provider=CLAUDE_PROVIDER,
    deepseek_api_key=DEEPSEEK_API_KEY,
    deepseek_api_url=DEEPSEEK_API_URL,
    deepseek_provider=DEEPSEEK_PROVIDER,
    ollama_api_url=OLLAMA_API_URL,
    is_origin_reasoning=IS_ORIGIN_REASONING
)

# 注册API路由
app.include_router(deepclaude_router)

# 验证日志级别
logger.debug("当前日志级别为 DEBUG")
logger.info("开始请求")

@app.get("/", dependencies=[Depends(verify_api_key)])
async def root():
    """根路径处理函数
    
    用于服务健康检查，需要API密钥验证
    
    Returns:
        dict: 包含欢迎消息的响应
    """
    logger.info("访问了根路径")
    return {"message": "Welcome to DeepClaude API"}

@app.get("/v1/models")
async def list_models():
    """获取支持的模型列表
    
    返回支持的模型信息，格式遵循OpenAI API标准
    不需要API密钥验证
    
    Returns:
        dict: 包含模型列表的响应，符合OpenAI API格式
    """
    models = [{
        "id": "deepclaude",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepclaude",
        "permission": [{
            "id": "modelperm-deepclaude",
            "object": "model_permission",
            "created": 1677610602,
            "allow_create_engine": False,
            "allow_sampling": True,
            "allow_logprobs": True,
            "allow_search_indices": False,
            "allow_view": True,
            "allow_fine_tuning": False,
            "organization": "*",
            "group": None,
            "is_blocking": False
        }],
        "root": "deepclaude",
        "parent": None
    }]
    
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    """处理聊天补全请求
    
    支持流式和非流式响应，兼容OpenAI和Claude API格式
    支持工具调用功能
    需要API密钥验证
    
    Args:
        request (Request): FastAPI请求对象
        
    Returns:
        StreamingResponse | JSONResponse: 流式或非流式响应
        
    Raises:
        HTTPException: 请求参数错误或服务器错误时抛出
    """
    try:
        # 记录原始请求信息
        raw_request = await request.json()
        logger.info("收到原始请求:")
        logger.info(f"请求头: {dict(request.headers)}")
        logger.info(f"请求体: {json.dumps(raw_request, ensure_ascii=False)}")
        
        # 验证必要参数
        if "messages" not in raw_request:
            raise ValueError("缺少必要的messages参数")
            
        # 提取工具相关参数，添加格式转换
        tools = raw_request.get("tools", [])
        
        # 检查并转换 OpenAI functions 格式到 Claude tools 格式
        if not tools and "functions" in raw_request:
            logger.info("检测到 OpenAI 格式的 functions 定义，正在转换为 tools 格式")
            functions = raw_request.get("functions", [])
            tools = []
            for func in functions:
                properties = func.get("parameters", {}).get("properties", {})
                required = func.get("parameters", {}).get("required", [])
                
                # 转换为Claude API兼容的格式
                tool = {
                    "name": func.get("name", "未命名工具"),
                    "description": func.get("description", ""),
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
                tools.append(tool)
                logger.info(f"转换工具: {func.get('name')} => {json.dumps(tool, ensure_ascii=False)}")
            
            logger.info(f"转换完成，共 {len(tools)} 个工具")
            raw_request["tools"] = tools
            
        tool_choice = raw_request.get("tool_choice", "auto")
        
        # 记录工具调用请求信息
        if tools:
            logger.info(f"收到工具调用请求，包含 {len(tools)} 个工具")
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool:
                        func = tool["function"]
                        logger.info(f"工具名称: {func.get('name', '未命名工具')}")
                        logger.info(f"工具描述: {func.get('description', '无描述')}")
                    elif "type" in tool and tool["type"] == "custom":
                        logger.info(f"工具名称: {tool.get('name', '未命名工具')}")
                        logger.info(f"工具描述: {tool.get('description', '无描述')}")
                    logger.debug(f"工具详情: {json.dumps(tool, ensure_ascii=False)}")
                else:
                    logger.warning(f"收到无效的工具定义: {tool}")
            
            # 记录工具选择策略
            if isinstance(tool_choice, str):
                logger.info(f"工具选择策略: {tool_choice}")
            elif isinstance(tool_choice, dict):
                logger.info(f"工具选择策略: {json.dumps(tool_choice, ensure_ascii=False)}")
            else:
                logger.info(f"工具选择策略: {tool_choice}")
            
        # 处理流式请求
        if raw_request.get("stream", False):
            logger.info("使用流式响应处理请求")
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=raw_request["messages"],
                    chat_id=f"chatcmpl-{uuid.uuid4()}",
                    created_time=int(time.time()),
                    model=raw_request.get("model", "deepclaude"),
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=raw_request.get("temperature", 0.7),
                    top_p=raw_request.get("top_p", 0.9),
                    presence_penalty=raw_request.get("presence_penalty", 0),
                    frequency_penalty=raw_request.get("frequency_penalty", 0)
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream;charset=utf-8",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                    "Keep-Alive": "timeout=600"
                }
            )
        
        # 处理非流式请求
        else:
            logger.info("使用非流式响应处理请求")
            model_args = get_and_validate_params(raw_request)
            response = await deep_claude.chat_completions_without_stream(
                messages=raw_request["messages"],
                model_arg=model_args,
                tools=tools,
                tool_choice=tool_choice,
                deepseek_model=raw_request.get("deepseek_model", "deepseek-reasoner"),
                claude_model=raw_request.get("model", os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219'))
            )
            
            # 记录工具调用响应
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "tool_calls" in choice.get("message", {}):
                    tool_calls = choice["message"]["tool_calls"]
                    logger.info(f"工具调用响应包含 {len(tool_calls)} 个工具调用")
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and "function" in tool_call:
                            func = tool_call["function"]
                            logger.info(f"调用工具: {func.get('name', '未知工具')}")
                            logger.debug(f"工具调用参数: {func.get('arguments', '{}')}")
                else:
                    logger.info("响应中不包含工具调用")
            
            return JSONResponse(content=response)
            
    except ValueError as e:
        logger.warning(f"参数验证错误: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }
        )
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "服务器内部错误",
                    "type": "server_error",
                    "param": None,
                    "code": None
                }
            }
        )

def get_and_validate_params(body: dict) -> tuple:
    """提取并验证请求参数
    
    从请求体中提取模型参数并进行验证
    
    Args:
        body (dict): 请求体字典
    
    Returns:
        tuple: (temperature, top_p, presence_penalty, frequency_penalty, stream)
    
    Raises:
        ValueError: 参数验证失败时抛出
    """
    try:
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))
        presence_penalty = float(body.get("presence_penalty", 0.0))
        frequency_penalty = float(body.get("frequency_penalty", 0.0))
        stream = bool(body.get("stream", True))

        # 验证参数范围
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature 必须在 0.0 到 2.0 之间")
            
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError("top_p 必须在 0.0 到 1.0 之间")
            
        if presence_penalty < -2.0 or presence_penalty > 2.0:
            raise ValueError("presence_penalty 必须在 -2.0 到 2.0 之间")
            
        if frequency_penalty < -2.0 or frequency_penalty > 2.0:
            raise ValueError("frequency_penalty 必须在 -2.0 到 2.0 之间")

        # Sonnet 模型特殊处理
        if "sonnet" in body.get("model", ""):
            if temperature < 0.0 or temperature > 1.0:
                raise ValueError("Sonnet 模型的 temperature 必须在 0.0 到 1.0 之间")

        return (temperature, top_p, presence_penalty, frequency_penalty, stream)
        
    except (TypeError, ValueError) as e:
        raise ValueError(f"参数验证失败: {str(e)}")

@app.post("/test_tool_call")
async def test_tool_call(request: Request):
    """测试工具调用功能的端点"""
    try:
        data = await request.json()
        messages = data.get("messages", [{"role": "user", "content": "今天北京天气怎么样？"}])
        
        # 定义测试工具
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，如北京、上海等"
                        },
                        "date": {
                            "type": "string", 
                            "description": "日期，默认为今天",
                            "enum": ["today", "tomorrow", "day_after_tomorrow"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        logger.info("开始测试工具调用 - 步骤1: 生成工具调用请求")
        
        # 第一步: 获取推理内容
        reasoning = await deep_claude._get_reasoning_content(
            messages=messages,
            model="deepseek-reasoner",
            model_arg=(0.7, 0.9, 0, 0)
        )
        
        logger.info(f"测试工具调用 - 步骤2: 获取到推理内容 ({len(reasoning)} 字符)")
        
        # 第二步: 获取工具调用决策
        original_question = messages[-1]["content"] if messages else ""
        decision_prompt = deep_claude._format_tool_decision_prompt(original_question, reasoning, tools)
        
        logger.info(f"测试工具调用 - 步骤3: 发送工具决策请求到Claude (提示长度: {len(decision_prompt)})")
        
        tool_decision = await deep_claude.claude_client.chat(
            messages=[{"role": "user", "content": decision_prompt}],
            tools=tools,
            tool_choice="auto",
            model=os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
        )
        
        logger.info(f"测试工具调用 - 步骤4: 收到Claude工具决策响应: {json.dumps(tool_decision)[:200]}...")
        
        # 第三步: 检查是否需要调用工具
        if "tool_calls" in tool_decision and tool_decision["tool_calls"]:
            tool_calls = tool_decision["tool_calls"]
            
            # 模拟工具调用
            logger.info(f"测试工具调用 - 步骤5: 决定使用 {len(tool_calls)} 个工具")
            
            # 构造工具响应
            tool_results = []
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = json.loads(func.get("arguments", "{}"))
                
                logger.info(f"测试工具调用 - 工具名称: {name}, 参数: {args}")
                
                # 模拟工具执行结果
                if name == "get_weather":
                    location = args.get("location", "")
                    date = args.get("date", "today")
                    result = {
                        "content": f"{location}今天天气晴朗，气温20-25度，适合外出活动。",
                        "tool_call_id": tool_call.get("id", "")
                    }
                    tool_results.append(result)
            
            # 第四步: 处理工具结果
            logger.info("测试工具调用 - 步骤6: 处理工具结果")
            final_answer = await deep_claude._handle_tool_results(
                original_question=original_question,
                reasoning=reasoning,
                tool_calls=tool_calls,
                tool_results=tool_results
            )
            
            logger.info("测试工具调用 - 步骤7: 生成最终回答")
            return {
                "success": True,
                "steps": {
                    "reasoning": reasoning,
                    "tool_decision": tool_decision,
                    "tool_results": tool_results,
                    "final_answer": final_answer
                }
            }
        else:
            logger.info("测试工具调用 - Claude决定不使用工具")
            return {
                "success": True,
                "message": "Claude决定不使用工具",
                "reasoning": reasoning,
                "tool_decision": tool_decision
            }
            
    except Exception as e:
        logger.error(f"工具调用测试失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
