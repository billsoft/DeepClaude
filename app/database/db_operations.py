"""
数据库操作类，封装常用的数据库操作

主要功能：
1. 用户管理 - 创建、查询用户
2. 对话会话管理 - 创建和更新对话会话
3. 对话历史记录 - 存储用户问题和AI回答
4. 数据统计 - 统计对话量和满意度等信息
"""
from sqlalchemy.exc import SQLAlchemyError
from .db_config import get_db_session, close_db_session
from .db_models import User, Role, Category, ConversationList, ConversationHistory, KnowledgeBase, RoleEnum, SatisfactionEnum
from app.utils.logger import logger
from typing import Optional, List, Dict, Any, Tuple
import datetime
import hashlib
import uuid

class DatabaseOperations:
    """数据库操作类"""
    
    @staticmethod
    def get_or_create_admin_user() -> int:
        """
        获取或创建默认管理员用户
        
        Returns:
            int: 用户ID
        """
        db = get_db_session()
        try:
            # 检查管理员角色是否存在
            admin_role = db.query(Role).filter(Role.name == "admin").first()
            if not admin_role:
                # 创建管理员角色
                admin_role = Role(name="admin", description="系统管理员，拥有所有权限")
                db.add(admin_role)
                db.commit()
                admin_role = db.query(Role).filter(Role.name == "admin").first()
            
            # 检查默认管理员用户是否存在
            admin_user = db.query(User).filter(User.username == "admin").first()
            if not admin_user:
                # 创建默认密码，使用SHA256加密
                default_password = hashlib.sha256("admin123".encode()).hexdigest()
                
                # 创建管理员用户
                admin_user = User(
                    username="admin",
                    password=default_password,
                    email="admin@deepsysai.com",
                    real_name="System Admin",
                    role_id=admin_role.id,
                    status=1
                )
                db.add(admin_user)
                db.commit()
                admin_user = db.query(User).filter(User.username == "admin").first()
            
            return admin_user.id
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"获取或创建管理员用户失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def create_conversation(user_id: Optional[int] = None, title: Optional[str] = None, 
                          category_id: Optional[int] = None) -> int:
        """
        创建新的对话会话
        
        Args:
            user_id: 用户ID，如果为None则使用管理员用户
            title: 对话标题，如果为None则使用默认标题
            category_id: 分类ID，默认为None
            
        Returns:
            int: 创建的对话ID
        """
        db = get_db_session()
        try:
            # 如果没有指定用户ID，使用管理员用户
            if user_id is None:
                user_id = DatabaseOperations.get_or_create_admin_user()
            
            # 如果没有指定标题，使用默认标题
            if title is None:
                title = f"对话_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 创建新对话
            conversation = ConversationList(
                user_id=user_id,
                title=title,
                category_id=category_id,
                is_completed=False
            )
            db.add(conversation)
            db.commit()
            return conversation.id
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"创建对话失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def add_conversation_history(conversation_id: int, user_id: Optional[int] = None, 
                               role: str = "user", content: str = "", reasoning: Optional[str] = None,
                               model_name: Optional[str] = None, tokens: Optional[int] = None) -> int:
        """
        添加对话历史记录
        
        Args:
            conversation_id: 对话ID
            user_id: 用户ID，如果为None则使用管理员用户
            role: 角色，user或ai
            content: 对话内容
            reasoning: 思考过程，仅当role为ai时有效
            model_name: 使用的模型名称，仅当role为ai时有效
            tokens: token数量，用于计算资源使用
            
        Returns:
            int: 创建的历史记录ID
        """
        db = get_db_session()
        try:
            # 如果没有指定用户ID，使用管理员用户
            if user_id is None:
                user_id = DatabaseOperations.get_or_create_admin_user()
            
            # 检查对话是否存在
            conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
            if not conversation:
                raise ValueError(f"对话ID {conversation_id} 不存在")
            
            # 创建对话历史
            history = ConversationHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                role=RoleEnum(role),
                content=content,
                reasoning=reasoning,
                model_name=model_name,
                tokens=tokens
            )
            db.add(history)
            
            # 更新对话最后更新时间
            conversation.update_time = datetime.datetime.now()
            
            db.commit()
            return history.id
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"添加对话历史失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def get_conversation_history(conversation_id: int) -> List[Dict[str, Any]]:
        """
        获取指定对话的历史记录
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        db = get_db_session()
        try:
            histories = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id
            ).order_by(ConversationHistory.create_time).all()
            
            result = []
            for history in histories:
                result.append({
                    "id": history.id,
                    "role": history.role.value,
                    "content": history.content,
                    "reasoning": history.reasoning,
                    "create_time": history.create_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": history.model_name,
                    "tokens": history.tokens
                })
            return result
        except SQLAlchemyError as e:
            logger.error(f"获取对话历史失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def complete_conversation(conversation_id: int, satisfaction: Optional[str] = None, 
                            feedback: Optional[str] = None) -> bool:
        """
        完成对话并添加评价
        
        Args:
            conversation_id: 对话ID
            satisfaction: 满意度评价，可以是"satisfied", "neutral", "unsatisfied"中的一个
            feedback: 反馈内容
            
        Returns:
            bool: 操作是否成功
        """
        db = get_db_session()
        try:
            conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
            if not conversation:
                raise ValueError(f"对话ID {conversation_id} 不存在")
            
            # 更新对话状态为已完成
            conversation.is_completed = True
            
            # 如果有评价，添加评价
            if satisfaction:
                conversation.satisfaction = SatisfactionEnum(satisfaction)
            
            # 如果有反馈，添加反馈
            if feedback:
                conversation.feedback = feedback
            
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"完成对话失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def get_user_conversations(user_id: Optional[int] = None, 
                             limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取用户的对话列表
        
        Args:
            user_id: 用户ID，如果为None则使用管理员用户
            limit: 返回结果数量限制
            offset: 分页偏移量
            
        Returns:
            List[Dict[str, Any]]: 对话列表
        """
        db = get_db_session()
        try:
            # 如果没有指定用户ID，使用管理员用户
            if user_id is None:
                user_id = DatabaseOperations.get_or_create_admin_user()
            
            conversations = db.query(ConversationList).filter(
                ConversationList.user_id == user_id
            ).order_by(ConversationList.update_time.desc()).limit(limit).offset(offset).all()
            
            result = []
            for conversation in conversations:
                # 获取最后一条AI回复作为预览
                last_ai_reply = db.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation.id,
                    ConversationHistory.role == RoleEnum.ai
                ).order_by(ConversationHistory.create_time.desc()).first()
                
                # 获取消息数量
                message_count = db.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation.id
                ).count()
                
                preview = last_ai_reply.content[:100] + "..." if last_ai_reply and len(last_ai_reply.content) > 100 else (
                    last_ai_reply.content if last_ai_reply else "")
                
                result.append({
                    "id": conversation.id,
                    "title": conversation.title,
                    "create_time": conversation.create_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "update_time": conversation.update_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_completed": conversation.is_completed,
                    "satisfaction": conversation.satisfaction.value if conversation.satisfaction else None,
                    "message_count": message_count,
                    "preview": preview
                })
            return result
        except SQLAlchemyError as e:
            logger.error(f"获取用户对话列表失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def update_conversation_title(conversation_id: int, title: str) -> bool:
        """
        更新对话标题
        
        Args:
            conversation_id: 对话ID
            title: 新标题
            
        Returns:
            bool: 操作是否成功
        """
        db = get_db_session()
        try:
            conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
            if not conversation:
                raise ValueError(f"对话ID {conversation_id} 不存在")
            
            conversation.title = title
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"更新对话标题失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def delete_conversation(conversation_id: int) -> bool:
        """
        删除对话及其历史记录
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            bool: 操作是否成功
        """
        db = get_db_session()
        try:
            conversation = db.query(ConversationList).filter(ConversationList.id == conversation_id).first()
            if not conversation:
                raise ValueError(f"对话ID {conversation_id} 不存在")
            
            # 删除对话历史记录
            db.query(ConversationHistory).filter(ConversationHistory.conversation_id == conversation_id).delete()
            
            # 删除对话
            db.delete(conversation)
            
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"删除对话失败: {e}")
            raise
        finally:
            close_db_session(db)
    
    @staticmethod
    def generate_conversation_id() -> str:
        """
        生成唯一的对话ID
        
        Returns:
            str: 唯一对话ID
        """
        return str(uuid.uuid4()) 