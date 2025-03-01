# 数据库更新记录

本文档记录了对数据库结构的所有更改，包括字段添加、表结构修改等。

## 2025-03-01：添加思考过程字段

### 变更内容

在 `conversation_history` 表中添加 `reasoning` 列，用于存储AI思考过程。

```sql
ALTER TABLE `conversation_history` 
ADD COLUMN `reasoning` TEXT NULL COMMENT '思考过程内容' AFTER `model_name`;
```

### 变更原因

为支持DeepClaude的思考和回答分离模式，需要存储AI的思考过程。原始数据库建表脚本中遗漏了该字段，但在应用代码的数据库模型中已经定义。

### 变更实现

1. 创建了 `app/database/db_utils.py` 模块，提供 `add_reasoning_column_if_not_exists()` 方法
2. 在测试运行前自动检查并添加缺失的字段
3. 更新了数据库表结构，保持与代码模型一致

### 相关文件

- `app/database/db_models.py` - 包含 ConversationHistory 模型定义
- `app/database/db_utils.py` - 提供数据库迁移工具
- `app/database/add_reasoning_column.sql` - SQL变更脚本
- `test/test_database.py` - 在测试前自动执行迁移

### 注意事项

- 此更改向后兼容，不会影响现有数据
- 新增列允许为NULL，因此不需要为现有记录添加默认值 