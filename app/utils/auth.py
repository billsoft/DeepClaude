# 导入所需的依赖
from fastapi import HTTPException, Header  # FastAPI相关组件
from typing import Optional  # 类型提示
import os  # 操作系统接口
from dotenv import load_dotenv  # 环境变量加载
from app.utils.logger import logger  # 日志记录器

# 初始化环境配置
# 加载 .env 文件，确保所有必要的环境变量都被正确加载
logger.info(f"当前工作目录: {os.getcwd()}")
logger.info("尝试加载.env文件...")
load_dotenv(override=True)  # 添加override=True强制覆盖已存在的环境变量，确保使用最新的配置

# 获取并验证API密钥环境变量
# ALLOW_API_KEY用于API访问认证，必须在环境变量中设置
ALLOW_API_KEY = os.getenv("ALLOW_API_KEY")
logger.info(f"ALLOW_API_KEY环境变量状态: {'已设置' if ALLOW_API_KEY else '未设置'}")

# 如果未设置API密钥，立即抛出异常，确保应用安全性
if not ALLOW_API_KEY:
    raise ValueError("ALLOW_API_KEY environment variable is not set")

# 打印API密钥的前4位用于调试
logger.info(f"Loaded API key starting with: {ALLOW_API_KEY[:4] if len(ALLOW_API_KEY) >= 4 else ALLOW_API_KEY}")


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """验证API密钥
    
    用于验证请求中的API密钥是否有效。该函数作为FastAPI的依赖项使用，
    用于保护API端点。它会检查请求头中的Authorization字段，验证其中的
    Bearer token是否与配置的API密钥匹配。

    Args:
        authorization (Optional[str], optional): Authorization header中的API密钥. 
            格式应为"Bearer <api_key>". Defaults to Header(None).

    Raises:
        HTTPException: 当Authorization header缺失或API密钥无效时抛出401错误
    """
    if authorization is None:
        logger.warning("请求缺少Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != ALLOW_API_KEY:
        logger.warning(f"无效的API密钥: {api_key}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    logger.info("API密钥验证通过")
