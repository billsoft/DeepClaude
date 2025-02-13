# 导入所需的Python标准库和第三方库
import logging  # 日志处理的核心库
import colorlog  # 用于生成彩色日志输出的库
import sys  # 系统相关功能
import os  # 操作系统接口
from dotenv import load_dotenv  # 用于加载.env环境变量文件

# 确保环境变量被加载
# 在程序启动时自动加载.env文件中的环境变量
load_dotenv()

def get_log_level() -> int:
    """从环境变量获取日志级别
    
    通过环境变量LOG_LEVEL获取日志级别，支持DEBUG、INFO、WARNING、ERROR、CRITICAL
    如果未设置或设置的值无效，默认返回INFO级别
    
    Returns:
        int: logging 模块定义的日志级别
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    return level_map.get(level, logging.INFO)

def setup_logger(name: str = "DeepClaude") -> logging.Logger:
    """设置一个彩色的logger
    
    配置一个支持彩色输出的日志记录器，包括以下特性：
    1. 支持不同级别日志的彩色显示
    2. 自定义日志格式，包含时间戳、logger名称、日志级别和消息
    3. 日志输出到标准输出(stdout)
    4. 避免重复创建handler

    Args:
        name (str, optional): logger的名称. Defaults to "DeepClaude".

    Returns:
        logging.Logger: 配置好的logger实例
    """
    logger = colorlog.getLogger(name)
    
    if logger.handlers:
        return logger
    
    # 从环境变量获取日志级别
    log_level = get_log_level()
    
    # 设置日志级别
    logger.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 设置彩色日志格式
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 创建一个默认的logger实例
# 这个logger实例将在整个应用程序中被其他模块导入和使用
logger = setup_logger()
