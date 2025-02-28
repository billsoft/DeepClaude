"""日志设置模块

该模块提供统一的日志配置，支持控制台输出与文件输出。
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import inspect

# 获取环境变量中的日志级别，默认为INFO
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

class DebuggableLogger(logging.Logger):
    """增强型日志记录器，添加更多调试功能"""
    
    def debug_stream(self, data, max_length=500):
        """记录流式数据，带有长度限制
        
        适用于记录大型流式响应的调试信息
        """
        if self.isEnabledFor(logging.DEBUG):
            data_str = str(data)
            if len(data_str) > max_length:
                data_str = data_str[:max_length] + "... [截断]"
            
            # 获取调用者信息
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            
            self.debug(f"[{filename}:{lineno}] 流式数据: {data_str}")
            
    def debug_response(self, response, max_length=300):
        """记录API响应内容
        
        适用于API调用的响应记录
        """
        if self.isEnabledFor(logging.DEBUG):
            resp_str = str(response)
            if len(resp_str) > max_length:
                resp_str = resp_str[:max_length] + "... [截断]"
            
            # 获取调用者信息
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            
            self.debug(f"[{filename}:{lineno}] API响应: {resp_str}")

# 注册自定义日志记录器类
logging.setLoggerClass(DebuggableLogger)

# 创建日志记录器
logger = logging.getLogger('deepclaude')

# 设置日志级别
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logger.setLevel(log_level)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)

# 设置日志格式
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(console_handler)

# 如果LOG_TO_FILE环境变量为true，启用文件日志
if os.getenv('LOG_TO_FILE', 'false').lower() == 'true':
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'deepclaude.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 添加到记录器
    logger.addHandler(file_handler)

# 打印初始化信息
logger.info(f"日志级别设置为: {LOG_LEVEL}")
if log_level <= logging.DEBUG:
    logger.debug("调试模式已开启")
