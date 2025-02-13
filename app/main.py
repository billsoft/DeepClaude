import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import logger
from app.utils.auth import verify_api_key
from app.deepclaude.deepclaude import DeepClaude

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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

# 是否使用原始推理格式，默认为True
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"

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

# 创建 DeepClaude 实例, 提出为Global变量
if not DEEPSEEK_API_KEY or not CLAUDE_API_KEY:
    logger.critical("请设置环境变量 CLAUDE_API_KEY 和 DEEPSEEK_API_KEY")
    sys.exit(1)

deep_claude = DeepClaude(
    DEEPSEEK_API_KEY,
    CLAUDE_API_KEY,
    DEEPSEEK_API_URL,
    CLAUDE_API_URL,
    CLAUDE_PROVIDER,
    IS_ORIGIN_REASONING
)

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
    """处理聊天完成请求
    
    支持流式和非流式输出，需要API密钥验证
    请求体格式遵循OpenAI API标准
    
    Args:
        request (Request): FastAPI请求对象，包含以下字段：
            - messages: 消息历史列表
            - model: 模型名称（可选）
            - stream: 是否使用流式输出（可选，默认True）
            - temperature: 采样温度（可选，默认0.5）
            - top_p: 核采样（可选，默认0.9）
            - presence_penalty: 主题新鲜度（可选，默认0.0）
            - frequency_penalty: 词频惩罚度（可选，默认0.0）
    
    Returns:
        Union[StreamingResponse, dict]: 
            - 流式输出时返回StreamingResponse对象
            - 非流式输出时返回包含回复内容的字典
    
    Raises:
        HTTPException: 请求参数验证失败时抛出
        Exception: 处理过程中的其他错误
    """

    try:
        # 1. 获取基础信息
        body = await request.json()
        messages = body.get("messages")

        # 2. 获取并验证参数
        model_arg = (
            get_and_validate_params(body)
        )
        stream = model_arg[4]  # 获取 stream 参数

        # 3. 根据 stream 参数返回相应的响应
        if stream:
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=messages,
                    model_arg=model_arg[:4],  # 不传递 stream 参数
                    deepseek_model=DEEPSEEK_MODEL,
                    claude_model=CLAUDE_MODEL
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式输出
            response = await deep_claude.chat_completions_without_stream(
                messages=messages,
                model_arg=model_arg[:4],  # 不传递 stream 参数
                deepseek_model=DEEPSEEK_MODEL,
                claude_model=CLAUDE_MODEL
            )
            return response

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        return {"error": str(e)}


def get_and_validate_params(body: dict) -> tuple:
    """提取并验证请求参数
    
    从请求体中提取模型参数并进行验证
    
    Args:
        body (dict): 请求体字典
    
    Returns:
        tuple: 包含以下参数的元组：
            - temperature: 采样温度（float）
            - top_p: 核采样（float）
            - presence_penalty: 主题新鲜度（float）
            - frequency_penalty: 词频惩罚度（float）
            - stream: 是否流式输出（bool）
    
    Raises:
        ValueError: 参数验证失败时抛出
    """
    # TODO: 默认值设定允许自定义
    temperature: float = body.get("temperature", 0.5)
    top_p: float = body.get("top_p", 0.9)
    presence_penalty: float = body.get("presence_penalty", 0.0)
    frequency_penalty: float = body.get("frequency_penalty", 0.0)
    stream: bool = body.get("stream", True)

    if "sonnet" in body.get("model", ""): # Only Sonnet 设定 temperature 必须在 0 到 1 之间
        if not isinstance(temperature, (float)) or temperature < 0.0 or temperature > 1.0:
            raise ValueError("Sonnet 设定 temperature 必须在 0 到 1 之间")

    return (temperature, top_p, presence_penalty, frequency_penalty, stream)
