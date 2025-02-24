# 导入系统相关模块
import os  # 用于操作系统相关功能，如环境变量读取
import sys  # 用于系统相关功能，如程序退出
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量
import uuid
import time

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
from app.deepclaude.deepclaude import DeepClaude  # DeepClaude核心类
from fastapi.responses import JSONResponse  # 用于返回JSON响应

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
# DEEPSEEK_PROVIDER: DeepSeek 提供商, 默认为 deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
DEEPSEEK_PROVIDER = os.getenv("DEEPSEEK_PROVIDER", "deepseek")

# 是否使用原始推理格式
IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "false").lower() == "true"

# 获取思考者提供商配置
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "deepseek").lower()

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

if not CLAUDE_API_KEY:
    logger.critical("必须设置 CLAUDE_API_KEY")
    sys.exit(1)

# 创建 DeepClaude 实例
deep_claude = DeepClaude(
    # Claude配置(回答者)
    claude_api_key=CLAUDE_API_KEY,
    claude_api_url=CLAUDE_API_URL,
    claude_provider=CLAUDE_PROVIDER,
    
    # DeepSeek配置(思考者选项1)
    deepseek_api_key=DEEPSEEK_API_KEY,
    deepseek_api_url=DEEPSEEK_API_URL,
    deepseek_provider=DEEPSEEK_PROVIDER,
    
    # Ollama配置(思考者选项2)
    ollama_api_url=OLLAMA_API_URL,
    
    # 思考格式配置
    is_origin_reasoning=IS_ORIGIN_REASONING
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
    try:
        data = await request.json()
        
        # 验证必要参数
        if "messages" not in data:
            raise ValueError("Missing messages parameter")
            
        if data.get("stream", False):
            # 设置完整的 SSE 响应头
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=data["messages"],
                    chat_id=f"chatcmpl-{uuid.uuid4()}",
                    created_time=int(time.time()),
                    model=data.get("model", "deepclaude")
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
        else:
            # 处理非流式请求
            response = await deep_claude.chat_completions_without_stream(
                messages=data["messages"],
                model_arg=get_and_validate_params(data)
            )
            return JSONResponse(content=response)
            
    except ValueError as e:
        logger.warning(f"参数验证错误: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

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
