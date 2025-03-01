"""
数据库配置模块，提供数据库连接和会话管理功能

主要功能：
1. 创建SQLAlchemy引擎和会话工厂
2. 提供获取和关闭数据库会话的工具函数
3. 定义ORM基类供模型继承
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from dotenv import load_dotenv
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

# 数据库连接配置
DB_URL = os.getenv("DB_URL", "mysql+pymysql://root:123654CCc.@bj-cdb-oqrn4mh2.sql.tencentcdb.com:24734/deepsysai?charset=utf8mb3")

# 创建SQLAlchemy引擎
engine = create_engine(
    DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True
)

# 创建会话工厂
SessionFactory = sessionmaker(bind=engine, autoflush=False)

# 创建线程安全的会话，在每个请求中可以安全使用
Session = scoped_session(SessionFactory)

# 创建声明式基类，所有模型类都将继承此类
Base = declarative_base()

def get_db_session():
    """创建并返回一个新的数据库会话"""
    db = Session()
    try:
        return db
    except Exception as e:
        db.rollback()
        logger.error(f"数据库会话创建失败: {e}")
        raise

def close_db_session(db):
    """关闭数据库会话"""
    db.close() 