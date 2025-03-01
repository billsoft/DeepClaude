#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库工具函数模块
提供数据库迁移、升级等功能
"""

import os
from sqlalchemy import text
from .db_config import get_db_session, close_db_session
from app.utils.logger import logger

def add_reasoning_column_if_not_exists():
    """
    检查并添加reasoning列到conversation_history表
    如果列已存在，则不做任何操作
    """
    db = get_db_session()
    try:
        # 先检查列是否存在
        check_sql = text("""
        SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'conversation_history' 
        AND COLUMN_NAME = 'reasoning'
        """)
        
        result = db.execute(check_sql).first()
        if result and result[0] > 0:
            logger.info("reasoning列已存在于conversation_history表中")
            return True
        
        # 添加列
        logger.info("正在向conversation_history表添加reasoning列...")
        add_column_sql = text("""
        ALTER TABLE `conversation_history` 
        ADD COLUMN `reasoning` TEXT NULL COMMENT '思考过程内容' AFTER `model_name`
        """)
        
        db.execute(add_column_sql)
        db.commit()
        
        # 验证是否添加成功
        verify_sql = text("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'conversation_history' 
        AND COLUMN_NAME = 'reasoning'
        """)
        
        verify_result = db.execute(verify_sql).first()
        if verify_result and verify_result[0] == 'reasoning':
            logger.info("reasoning列已成功添加到conversation_history表中")
            return True
        else:
            logger.error("添加reasoning列失败，请检查数据库权限")
            return False
            
    except Exception as e:
        db.rollback()
        logger.error(f"添加reasoning列时发生错误: {e}")
        return False
    finally:
        close_db_session(db) 