"""日志处理模块"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import time
import json

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def formatException(self, exc_info):
        """格式化异常信息"""
        result = super().formatException(exc_info)
        return result
    
    def format(self, record):
        """格式化日志记录"""
        try:
            # 截断过长的消息
            if record.levelno == logging.DEBUG and len(record.msg) > 10000:
                record.msg = record.msg[:10000] + "... [截断]"
            
            # 如果是JSON对象转字符串，并尝试格式化
            if isinstance(record.msg, dict):
                try:
                    record.msg = json.dumps(record.msg, ensure_ascii=False, indent=2)
                except:
                    pass
        except:
            pass
            
        return super().format(record)

# 创建日志处理器
def setup_logger(name='deepclaude'):
    """设置日志处理器"""
    
    # 确保日志目录存在
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间作为日志文件名
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
    
    # 创建日志处理器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)
    
    # 如果已有处理器则不重新添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # 创建文件处理器
    max_bytes = 10 * 1024 * 1024  # 10 MB
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    
    # 设置日志格式
    formatter = CustomFormatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"日志级别设置为: {log_level}")
    
    return logger

# 创建全局日志记录器
logger = setup_logger()
