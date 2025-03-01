#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库操作单元测试

测试数据库操作的所有功能，包括：
1. 管理员用户的创建与获取
2. 对话会话的创建、查询和删除
3. 对话历史记录的添加与查询
4. 对话满意度评价
5. 用户对话列表查询
6. 对话标题更新
7. 测试完成后清理所有测试数据
"""

import os
import sys
import unittest
import datetime
import hashlib
import time
from dotenv import load_dotenv
from sqlalchemy import text

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入需要测试的模块
from app.database.db_config import get_db_session, close_db_session
from app.database.db_models import User, Role, ConversationList, ConversationHistory, RoleEnum, SatisfactionEnum
from app.database.db_operations import DatabaseOperations
from app.database.db_utils import add_reasoning_column_if_not_exists
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

class TestDatabaseOperations(unittest.TestCase):
    """数据库操作测试类"""
    
    # 存储测试中创建的数据ID
    test_data = {
        "admin_user_id": None,
        "test_user_id": None,
        "conversation_id": None,
        "history_user_id": None,
        "history_ai_id": None,
    }
    
    @classmethod
    def setUpClass(cls):
        """在所有测试之前设置测试环境"""
        logger.info("======== 开始数据库操作测试 ========")
        # 验证数据库连接
        db = get_db_session()
        try:
            # 检查连接是否正常
            db.execute(text("SELECT 1"))
            logger.info("数据库连接正常")
            
            # 确保数据库表结构与模型一致
            logger.info("检查并更新数据库表结构...")
            add_result = add_reasoning_column_if_not_exists()
            if add_result:
                logger.info("数据库表结构检查完成，确保conversation_history表包含reasoning列")
            else:
                logger.warning("数据库表结构更新失败，测试可能会失败")
        except Exception as e:
            logger.error(f"数据库连接或结构更新失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @classmethod
    def tearDownClass(cls):
        """在所有测试之后清理测试环境"""
        logger.info("======== 数据库操作测试完成 ========")
    
    def test_01_get_or_create_admin_user(self):
        """测试获取或创建管理员用户"""
        logger.info("测试获取或创建管理员用户")
        try:
            admin_id = DatabaseOperations.get_or_create_admin_user()
            self.assertIsNotNone(admin_id, "管理员用户ID不应为空")
            
            # 存储管理员ID，后续测试使用
            self.__class__.test_data["admin_user_id"] = admin_id
            logger.info(f"管理员用户ID: {admin_id}")
            
            # 查询验证用户信息
            db = get_db_session()
            try:
                admin_user = db.query(User).filter(User.id == admin_id).first()
                self.assertIsNotNone(admin_user, "管理员用户应该存在")
                self.assertEqual(admin_user.username, "admin", "管理员用户名应为admin")
                
                # 验证角色
                admin_role = db.query(Role).filter(Role.name == "admin").first()
                self.assertIsNotNone(admin_role, "管理员角色应该存在")
                self.assertEqual(admin_user.role_id, admin_role.id, "用户应关联到管理员角色")
            finally:
                close_db_session(db)
        except Exception as e:
            self.fail(f"获取或创建管理员用户时发生错误: {e}")
    
    def test_02_create_test_user(self):
        """创建测试用户"""
        logger.info("测试创建测试用户")
        db = get_db_session()
        try:
            # 确保admin角色存在
            admin_role = db.query(Role).filter(Role.name == "admin").first()
            self.assertIsNotNone(admin_role, "管理员角色应该存在")
            
            # 创建一个测试用户
            timestamp = int(time.time())
            test_username = f"test_user_{timestamp}"
            test_password = hashlib.sha256(f"test_password_{timestamp}".encode()).hexdigest()
            
            test_user = User(
                username=test_username,
                password=test_password,
                email=f"test_{timestamp}@example.com",
                real_name="Test User",
                role_id=admin_role.id,
                status=1
            )
            db.add(test_user)
            db.commit()
            
            # 获取创建的用户ID
            created_user = db.query(User).filter(User.username == test_username).first()
            self.assertIsNotNone(created_user, "测试用户应该存在")
            
            # 存储测试用户ID
            self.__class__.test_data["test_user_id"] = created_user.id
            logger.info(f"测试用户ID: {created_user.id}")
        except Exception as e:
            db.rollback()
            self.fail(f"创建测试用户时发生错误: {e}")
        finally:
            close_db_session(db)
    
    def test_03_create_conversation(self):
        """测试创建对话会话"""
        logger.info("测试创建对话会话")
        try:
            # 使用测试用户ID创建对话
            user_id = self.__class__.test_data["test_user_id"]
            title = f"测试对话_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            conversation_id = DatabaseOperations.create_conversation(
                user_id=user_id,
                title=title
            )
            
            self.assertIsNotNone(conversation_id, "对话ID不应为空")
            
            # 存储对话ID
            self.__class__.test_data["conversation_id"] = conversation_id
            logger.info(f"创建的对话ID: {conversation_id}")
            
            # 验证对话信息
            db = get_db_session()
            try:
                conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
                self.assertIsNotNone(conversation, "对话应该存在")
                self.assertEqual(conversation.title, title, "对话标题应匹配")
                self.assertEqual(conversation.user_id, user_id, "对话应该关联到测试用户")
                self.assertFalse(conversation.is_completed, "新创建的对话应该是未完成状态")
            finally:
                close_db_session(db)
        except Exception as e:
            self.fail(f"创建对话会话时发生错误: {e}")
    
    def test_04_add_user_question(self):
        """测试添加用户问题"""
        logger.info("测试添加用户问题")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            user_id = self.__class__.test_data["test_user_id"]
            content = "这是一个测试问题，DeepClaude如何工作？"
            
            history_id = DatabaseOperations.add_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=content
            )
            
            self.assertIsNotNone(history_id, "历史记录ID不应为空")
            
            # 存储历史记录ID
            self.__class__.test_data["history_user_id"] = history_id
            logger.info(f"用户问题历史记录ID: {history_id}")
            
            # 验证历史记录信息
            db = get_db_session()
            try:
                history = db.query(ConversationHistory).filter(ConversationHistory.id == history_id).first()
                self.assertIsNotNone(history, "历史记录应该存在")
                self.assertEqual(history.role, RoleEnum.user, "角色应为用户")
                self.assertEqual(history.content, content, "内容应匹配")
                self.assertEqual(history.conversation_id, conversation_id, "历史记录应该关联到对话")
                self.assertEqual(history.user_id, user_id, "历史记录应该关联到用户")
            finally:
                close_db_session(db)
        except Exception as e:
            self.fail(f"添加用户问题时发生错误: {e}")
    
    def test_05_add_ai_answer(self):
        """测试添加AI回答"""
        logger.info("测试添加AI回答")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            user_id = self.__class__.test_data["test_user_id"]
            content = "DeepClaude是一个集成了DeepSeek和Claude两个大语言模型能力的服务，它的工作流程是：1. 使用DeepSeek进行思考，2. 将思考结果传递给Claude生成最终答案。"
            reasoning = "我需要解释DeepClaude是什么以及它如何工作。DeepClaude实际上是一个结合了多个模型能力的服务，它的独特之处在于将推理和生成分开..."
            model_name = "claude-3-7-sonnet-20250219"
            tokens = 256
            
            history_id = DatabaseOperations.add_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                role="ai",
                content=content,
                reasoning=reasoning,
                model_name=model_name,
                tokens=tokens
            )
            
            self.assertIsNotNone(history_id, "历史记录ID不应为空")
            
            # 存储历史记录ID
            self.__class__.test_data["history_ai_id"] = history_id
            logger.info(f"AI回答历史记录ID: {history_id}")
            
            # 验证历史记录信息
            db = get_db_session()
            try:
                history = db.query(ConversationHistory).filter(ConversationHistory.id == history_id).first()
                self.assertIsNotNone(history, "历史记录应该存在")
                self.assertEqual(history.role, RoleEnum.ai, "角色应为AI")
                self.assertEqual(history.content, content, "内容应匹配")
                self.assertEqual(history.reasoning, reasoning, "思考过程应匹配")
                self.assertEqual(history.model_name, model_name, "模型名称应匹配")
                self.assertEqual(history.tokens, tokens, "Token数量应匹配")
            finally:
                close_db_session(db)
        except Exception as e:
            self.fail(f"添加AI回答时发生错误: {e}")
    
    def test_06_get_conversation_history(self):
        """测试获取对话历史"""
        logger.info("测试获取对话历史")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            
            histories = DatabaseOperations.get_conversation_history(conversation_id)
            
            self.assertIsNotNone(histories, "历史记录列表不应为空")
            self.assertEqual(len(histories), 2, "应该有2条历史记录")
            
            # 验证历史记录内容
            user_history = next((h for h in histories if h["role"] == "user"), None)
            ai_history = next((h for h in histories if h["role"] == "ai"), None)
            
            self.assertIsNotNone(user_history, "用户历史记录应存在")
            self.assertIsNotNone(ai_history, "AI历史记录应存在")
            
            # 验证用户历史记录
            self.assertEqual(user_history["id"], self.__class__.test_data["history_user_id"], "用户历史记录ID应匹配")
            
            # 验证AI历史记录
            self.assertEqual(ai_history["id"], self.__class__.test_data["history_ai_id"], "AI历史记录ID应匹配")
            self.assertIsNotNone(ai_history["reasoning"], "AI历史记录应包含思考过程")
            
            logger.info("成功获取对话历史记录")
        except Exception as e:
            self.fail(f"获取对话历史时发生错误: {e}")
    
    def test_07_update_conversation_title(self):
        """测试更新对话标题"""
        logger.info("测试更新对话标题")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            new_title = f"更新后的标题_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            result = DatabaseOperations.update_conversation_title(
                conversation_id=conversation_id,
                title=new_title
            )
            
            self.assertTrue(result, "更新对话标题应该成功")
            
            # 验证标题已更新
            db = get_db_session()
            try:
                conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
                self.assertEqual(conversation.title, new_title, "对话标题应已更新")
            finally:
                close_db_session(db)
                
            logger.info(f"对话标题已更新为: {new_title}")
        except Exception as e:
            self.fail(f"更新对话标题时发生错误: {e}")
    
    def test_08_add_satisfaction_feedback(self):
        """测试添加满意度评价"""
        logger.info("测试添加满意度评价")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            satisfaction = "satisfied"
            feedback = "这是一个很好的回答，解释得很清楚！"
            
            result = DatabaseOperations.complete_conversation(
                conversation_id=conversation_id,
                satisfaction=satisfaction,
                feedback=feedback
            )
            
            self.assertTrue(result, "添加满意度评价应该成功")
            
            # 验证满意度评价已添加
            db = get_db_session()
            try:
                conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
                self.assertEqual(conversation.satisfaction, SatisfactionEnum.satisfied, "满意度评价应为satisfied")
                self.assertEqual(conversation.feedback, feedback, "反馈内容应匹配")
                self.assertTrue(conversation.is_completed, "对话应标记为已完成")
            finally:
                close_db_session(db)
                
            logger.info("已成功添加满意度评价")
        except Exception as e:
            self.fail(f"添加满意度评价时发生错误: {e}")
    
    def test_09_get_user_conversations(self):
        """测试获取用户对话列表"""
        logger.info("测试获取用户对话列表")
        try:
            user_id = self.__class__.test_data["test_user_id"]
            
            conversations = DatabaseOperations.get_user_conversations(
                user_id=user_id,
                limit=10,
                offset=0
            )
            
            self.assertIsNotNone(conversations, "对话列表不应为空")
            self.assertGreaterEqual(len(conversations), 1, "应该至少有1个对话")
            
            # 找到我们创建的对话
            test_conversation = next((c for c in conversations if c["id"] == self.__class__.test_data["conversation_id"]), None)
            self.assertIsNotNone(test_conversation, "测试对话应该存在于列表中")
            
            # 验证对话信息
            self.assertTrue(test_conversation["is_completed"], "对话应标记为已完成")
            self.assertEqual(test_conversation["satisfaction"], "satisfied", "满意度评价应为satisfied")
            self.assertGreaterEqual(test_conversation["message_count"], 2, "消息数量应至少为2")
            
            logger.info("成功获取用户对话列表")
        except Exception as e:
            self.fail(f"获取用户对话列表时发生错误: {e}")
    
    def test_10_delete_conversation(self):
        """测试删除对话及其历史记录"""
        logger.info("测试删除对话及其历史记录")
        try:
            conversation_id = self.__class__.test_data["conversation_id"]
            
            result = DatabaseOperations.delete_conversation(conversation_id)
            
            self.assertTrue(result, "删除对话应该成功")
            
            # 验证对话已删除
            db = get_db_session()
            try:
                conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
                self.assertIsNone(conversation, "对话应该已被删除")
                
                # 验证历史记录已删除
                histories = db.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation_id
                ).all()
                self.assertEqual(len(histories), 0, "历史记录应该已被删除")
            finally:
                close_db_session(db)
                
            logger.info("对话及其历史记录已成功删除")
        except Exception as e:
            self.fail(f"删除对话时发生错误: {e}")
    
    def test_11_cleanup_test_user(self):
        """清理测试用户"""
        logger.info("清理测试用户")
        db = get_db_session()
        try:
            # 删除测试用户
            test_user_id = self.__class__.test_data["test_user_id"]
            db.query(User).filter(User.id == test_user_id).delete()
            db.commit()
            
            # 验证用户已删除
            test_user = db.query(User).filter(User.id == test_user_id).first()
            self.assertIsNone(test_user, "测试用户应该已被删除")
            
            logger.info("测试用户已成功清理")
        except Exception as e:
            db.rollback()
            self.fail(f"清理测试用户时发生错误: {e}")
        finally:
            close_db_session(db)
    
    def test_12_verify_cleanup(self):
        """验证所有测试数据已清理"""
        logger.info("验证所有测试数据已清理")
        db = get_db_session()
        try:
            # 验证对话已删除
            conversation_id = self.__class__.test_data["conversation_id"]
            conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
            self.assertIsNone(conversation, "对话应该已被删除")
            
            # 验证历史记录已删除
            history_user_id = self.__class__.test_data["history_user_id"]
            history_ai_id = self.__class__.test_data["history_ai_id"]
            
            user_history = db.query(ConversationHistory).filter(ConversationHistory.id == history_user_id).first()
            ai_history = db.query(ConversationHistory).filter(ConversationHistory.id == history_ai_id).first()
            
            self.assertIsNone(user_history, "用户历史记录应该已被删除")
            self.assertIsNone(ai_history, "AI历史记录应该已被删除")
            
            # 验证测试用户已删除
            test_user_id = self.__class__.test_data["test_user_id"]
            test_user = db.query(User).filter(User.id == test_user_id).first()
            self.assertIsNone(test_user, "测试用户应该已被删除")
            
            logger.info("所有测试数据已成功清理")
        except Exception as e:
            self.fail(f"验证数据清理时发生错误: {e}")
        finally:
            close_db_session(db)

if __name__ == "__main__":
    unittest.main() 