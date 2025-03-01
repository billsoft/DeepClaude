"""
数据库模型定义模块，包含所有表结构的ORM映射

主要表结构：
1. 用户表(users)和角色表(roles) - 用户和权限管理
2. 分类表(categories) - 对话和知识条目分类
3. 对话列表表(conversation_lists) - 管理对话会话
4. 历史对话表(conversation_history) - 记录具体问答内容
5. 知识库表(knowledge_base) - 存储提取的知识条目
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db_config import Base
import enum

# 定义满意度枚举类型
class SatisfactionEnum(enum.Enum):
    satisfied = "satisfied"
    neutral = "neutral"
    unsatisfied = "unsatisfied"

# 定义角色枚举类型
class RoleEnum(enum.Enum):
    user = "user"
    ai = "ai"

class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="用户ID，主键")
    username = Column(String(50), nullable=False, unique=True, comment="用户名")
    password = Column(String(255), nullable=False, comment="密码（加密存储）")
    email = Column(String(100), unique=True, nullable=True, comment="用户邮箱")
    real_name = Column(String(50), nullable=True, comment="用户真实姓名")
    phone = Column(String(20), nullable=True, comment="联系电话")
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, comment="角色ID，外键")
    refresh_token = Column(String(500), nullable=True, comment="JWT刷新令牌")
    token_expire_time = Column(DateTime, nullable=True, comment="令牌过期时间")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    last_login = Column(DateTime, nullable=True, comment="最后登录时间")
    status = Column(Integer, default=1, nullable=False, comment="状态：1-正常，0-禁用")
    avatar = Column(String(255), nullable=True, comment="用户头像URL")
    login_ip = Column(String(50), nullable=True, comment="最后登录IP")
    
    # 关系
    role = relationship("Role", back_populates="users")
    conversation_lists = relationship("ConversationList", back_populates="user")
    conversation_histories = relationship("ConversationHistory", back_populates="user")

class Role(Base):
    """角色表"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="角色ID，主键")
    name = Column(String(50), nullable=False, unique=True, comment="角色名称")
    description = Column(String(200), nullable=True, comment="角色描述")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    
    # 关系
    users = relationship("User", back_populates="role")

class Category(Base):
    """分类表"""
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="分类ID，主键")
    name = Column(String(100), nullable=False, comment="分类名称")
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="父分类ID，为空表示顶级分类")
    description = Column(String(500), nullable=True, comment="分类描述")
    sort_order = Column(Integer, default=0, nullable=False, comment="排序顺序")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    
    # 关系
    children = relationship("Category", back_populates="parent", remote_side=[id])
    parent = relationship("Category", back_populates="children", remote_side=[parent_id])
    conversation_lists = relationship("ConversationList", back_populates="category")
    knowledge_bases = relationship("KnowledgeBase", back_populates="category")

class ConversationList(Base):
    """对话列表表"""
    __tablename__ = "conversation_lists"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="对话列表ID，主键")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID，外键")
    title = Column(String(200), nullable=True, comment="对话标题，可自动生成或用户自定义")
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="分类ID，外键")
    satisfaction = Column(Enum(SatisfactionEnum), nullable=True, comment="用户满意度评价")
    feedback = Column(Text, nullable=True, comment="用户反馈内容")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    is_completed = Column(Boolean, default=False, nullable=False, comment="是否已完成：0-进行中，1-已完成")
    
    # 关系
    user = relationship("User", back_populates="conversation_lists")
    category = relationship("Category", back_populates="conversation_lists")
    conversation_histories = relationship("ConversationHistory", back_populates="conversation_list", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="source_conversation")

class ConversationHistory(Base):
    """历史对话表"""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="历史记录ID，主键")
    conversation_id = Column(Integer, ForeignKey("conversation_lists.id"), nullable=False, comment="所属对话列表ID，外键")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID，外键")
    role = Column(Enum(RoleEnum), nullable=False, comment="发言角色：用户或AI")
    content = Column(Text, nullable=False, comment="对话内容")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    is_error = Column(Boolean, default=False, nullable=False, comment="是否包含错误：0-正常，1-错误")
    is_duplicate = Column(Boolean, default=False, nullable=False, comment="是否重复内容：0-不是，1-是")
    tokens = Column(Integer, nullable=True, comment="Token数量，用于计算资源使用")
    model_name = Column(String(100), nullable=True, comment="使用的AI模型名称")
    reasoning = Column(Text, nullable=True, comment="思考过程内容")
    
    # 关系
    conversation_list = relationship("ConversationList", back_populates="conversation_histories")
    user = relationship("User", back_populates="conversation_histories")

class KnowledgeBase(Base):
    """知识库表"""
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="知识条目ID，主键")
    question = Column(String(500), nullable=False, comment="标准问题")
    answer = Column(Text, nullable=False, comment="标准答案")
    source_conversation_id = Column(Integer, ForeignKey("conversation_lists.id"), nullable=True, comment="来源对话ID，可为空")
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True, comment="分类ID，外键")
    keywords = Column(String(500), nullable=True, comment="关键词，用于检索")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")
    creator_id = Column(Integer, nullable=True, comment="创建者ID，可能是自动提取或人工创建")
    status = Column(Integer, default=1, nullable=False, comment="状态：1-启用，0-禁用")
    confidence_score = Column(Float, nullable=True, comment="置信度分数，表示该知识条目的可靠性")
    
    # 关系
    source_conversation = relationship("ConversationList", back_populates="knowledge_bases")
    category = relationship("Category", back_populates="knowledge_bases") 