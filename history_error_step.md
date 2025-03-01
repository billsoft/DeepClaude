# 错误步骤记录

## 2025-03-01：数据库模型与表结构不一致问题

### 错误信息
- **现象**：数据库测试失败，出现 "Unknown column 'reasoning' in 'field list'" 错误
- **错误日志**：
  ```
  pymysql.err.OperationalError: (1054, "Unknown column 'reasoning' in 'field list'")
  [SQL: INSERT INTO conversation_history (conversation_id, user_id, `role`, content, create_time, 
  is_error, is_duplicate, tokens, model_name, reasoning) VALUES ...]
  ```
- **影响范围**：所有涉及conversation_history表的操作，包括添加对话历史、查询历史记录等功能

### 错误分析
- **根本原因**：数据库建表脚本中没有定义reasoning字段，但SQLAlchemy模型中定义了该字段，造成模型与实际表结构不一致
- **尝试的方案**：
  1. 考虑修改测试代码，移除reasoning字段：
     - 结果：不可行
     - 原因：reasoning字段在实际业务中有用，用于存储AI的思考过程
  2. 修改数据库表结构，添加reasoning字段：
     - 结果：成功
     - 验证：测试全部通过

### 解决方案
- **最终方案**：创建数据库工具函数，自动检查并添加缺失的列
  ```python
  def add_reasoning_column_if_not_exists():
      # 检查列是否存在
      # 如果不存在则添加
      # 验证添加结果
  ```
- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：低
  - 资源消耗：低
  - 稳定性：高
  - 兼容性：高

### 经验总结
- **避免方向**：避免模型定义与数据库结构脱节，不应直接假设表结构已与模型一致
- **正确思路**：
  1. 数据库模型变更时同步更新建表脚本
  2. 开发迁移工具检查并修复表结构与模型不一致问题
  3. 在关键启动点（如测试前）检查数据库结构
- **最佳实践**：
  1. 维护数据库变更记录文档
  2. 实现自动化迁移工具
  3. 在应用启动前自动检查表结构
- **预防措施**：
  1. 使用版本化的数据库迁移工具（如Alembic）
  2. 在CI/CD流程中添加数据库结构验证步骤
  3. 开发与测试环境保持同步 