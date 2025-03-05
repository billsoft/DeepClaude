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

## 2024-03-01：工具调用实现问题 [P1]

### 错误信息
- **现象**：工具调用功能无法正常工作，Claude API 没有返回工具调用响应
- **错误日志**：
  ```
  工具调用决策结果: 不使用工具
  Claude API错误: 工具调用参数未正确传递
  ```
- **影响范围**：所有涉及工具调用的功能，包括天气查询、搜索等功能

### 错误分析
- **根本原因**：
  1. Claude API 版本过低，不支持工具调用功能
  2. 工具参数在传递过程中丢失
  3. 工具调用请求格式不符合 Claude API 要求
  4. 工具调用响应处理逻辑不完整

- **尝试的方案**：
  1. 直接传递工具参数到 Claude API：
     - 结果：失败
     - 原因：API 版本不支持工具调用
  
  2. 更新 API 版本但不修改请求格式：
     - 结果：失败
     - 原因：请求格式不符合新版本要求

  3. 完整实现工具调用流程：
     - 结果：成功
     - 验证：测试通过，能正确处理工具调用

### 解决方案
- **最终方案**：
  1. 更新 Claude API 版本到支持工具调用的版本
  ```python
  headers.update({
      "anthropic-version": "2024-02-15",
      "anthropic-beta": "messages-2024-02-15"
  })
  ```
  
  2. 在请求数据中正确添加工具参数
  ```python
  if "tools" in kwargs and kwargs["tools"]:
      data["tools"] = tools
      data["tool_choice"] = tool_choice
  ```
  
  3. 完善工具调用响应处理
  ```python
  if response['type'] == 'tool_calls':
      tool_calls = response.get('tool_calls', [])
      for tool_call in tool_calls:
          yield "tool_call", tool_call
  ```

- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：中
  - 资源消耗：低
  - 稳定性：高
  - 兼容性：高

### 经验总结
- **避免方向**：
  1. 不要假设 API 版本自动支持新功能
  2. 不要忽略工具参数的验证
  3. 不要在传递过程中丢失参数
  4. 不要忽略错误处理和日志记录

- **正确思路**：
  1. 确保 API 版本支持目标功能
  2. 完整实现工具调用流程
  3. 添加参数验证和错误处理
  4. 增加详细的日志记录

- **最佳实践**：
  1. 在工具调用前验证工具格式
  2. 记录工具调用的完整过程
  3. 实现优雅的错误处理
  4. 保持与 OpenAI API 格式兼容

- **预防措施**：
  1. 添加工具格式验证
  2. 实现完整的日志记录
  3. 添加工具调用测试用例
  4. 定期验证 API 版本兼容性

### 代码示例
- **问题代码**：
```python
# 直接传递未验证的工具参数
data["tools"] = kwargs.get("tools")
```

- **修复代码**：
```python
# 验证并处理工具参数
if "tools" in kwargs and kwargs["tools"]:
    tools = kwargs["tools"]
    for tool in tools:
        if not isinstance(tool, dict) or "function" not in tool:
            logger.warning(f"跳过无效的工具定义: {tool}")
            continue
    data["tools"] = tools
    logger.info(f"添加 {len(tools)} 个工具到请求")
```

- **关键差异**：
  1. 添加了工具格式验证
  2. 增加了日志记录
  3. 优化了错误处理
  4. 完善了参数传递 

#### **2024-03-19 15:30:00 - [TOOL-001] 工具调用格式转换和验证问题 [P2]**
- **标签**：#工具调用 #格式转换 #API兼容性
- **参考**：PARAM-001, API-103

##### **错误信息**
- **现象**：工具调用请求在传递过程中可能丢失或格式不正确
- **影响范围**：所有依赖工具调用的功能
- **复现条件**：
  1. 通过 Dify 发送带工具的请求
  2. 请求中的工具参数未正确传递到 DeepClaude

##### **错误分析**
- **根本原因**：
  1. 工具格式转换可能不完全兼容
  2. 工具验证逻辑可能过于严格
  3. 日志记录不够完整，难以追踪问题

- **尝试的方案**：
  1. 添加详细的请求数据日志：成功
  2. 实现工具格式转换功能：部分成功
  3. 增加工具验证和转换日志：成功

##### **解决方案**
- **最终方案**：
  1. 在入口处添加完整的请求数据日志
  2. 实现工具格式转换功能
  3. 增加工具验证和转换的详细日志
  4. 优化工具验证逻辑

- **评估**：
  - 有效性：部分解决（80%）
  - 复杂度：中
  - 资源消耗：低
  - 稳定性：高
  - 兼容性：中

##### **经验总结**
- **避免方向**：
  1. 避免过于严格的工具验证
  2. 避免在日志中省略关键信息
  3. 避免直接修改工具格式而不验证

- **正确思路**：
  1. 完整记录请求和响应数据
  2. 在关键节点添加详细日志
  3. 实现灵活的工具格式转换
  4. 保持向后兼容性

- **最佳实践**：
  1. 在入口处记录完整请求数据
  2. 在格式转换前后记录工具数据
  3. 使用统一的工具格式规范
  4. 提供详细的错误信息

##### **预防措施**
1. 定期检查日志完整性
2. 测试不同格式的工具调用
3. 验证工具格式转换结果
4. 保持与上游 API 的兼容性 

#### **2025-03-03 17:25:00 - [API-105] Claude API工具类型格式不兼容 [P0]**
- **标签**：#工具调用 #API格式 #Claude接口
- **参考**：TOOL-001, API-104

##### **错误信息**
- **现象**：Claude API请求返回400错误，提示tools.0参数格式不正确，不认可function类型
- **错误日志**：
  ```
  HTTP 400: {"type":"error","error":{"type":"invalid_request_error","message":"tools.0: Input tag 'function' found using 'type' does not match any of the expected tags: 'bash_20250124', 'custom', 'text_editor_20250124'"}}
  ```
- **影响范围**：所有使用工具调用功能的接口
- **复现条件**：使用OpenAI格式发送包含工具的请求到Claude API

##### **错误分析**
- **根本原因**：Claude API不接受OpenAI的`function`工具类型，只支持特定的工具类型标签：`bash_20250124`、`custom`、`text_editor_20250124`
- **尝试的方案**：
  1. 检查Claude API文档对比工具类型格式差异：
     - 结果：发现不匹配
     - 原因：Claude API不支持`function`作为工具类型，需要使用`custom`类型
  2. 修改工具格式转换逻辑：
     - 结果：成功
     - 验证：API调用成功，不再返回400错误

##### **解决方案**
- **最终方案**：
  1. 更新工具格式转换逻辑，将OpenAI的function格式转换为Claude支持的custom格式
  ```python
  # 将OpenAI格式工具转换为Claude格式
  tool = {
      "type": "custom",
      "name": func.get("name", "未命名工具"),
      "description": func.get("description", ""),
      "tool_schema": func.get("parameters", {})
  }
  ```
  2. 在ClaudeClient和DeepClaude类中添加工具格式验证和转换
  3. 在main.py中直接从OpenAI格式转换为Claude格式
- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：低
  - 资源消耗：低
  - 稳定性：高
  - 兼容性：高

##### **经验总结**
- **避免方向**：
  1. 避免假设不同API之间的工具格式兼容
  2. 避免在工具类型上直接使用`function`类型而不转换
  3. 避免忽略API错误信息中的具体要求

- **正确思路**：
  1. 查阅最新API文档，确认支持的工具类型
  2. 实现完整的工具格式转换
  3. 在转换前后添加详细日志
  4. 优先使用API原生支持的工具类型

- **最佳实践**：
  1. 记录API请求的完整工具参数和格式
  2. 在格式转换前后记录工具格式变化
  3. 注意不同API之间工具类型的差异
  4. 优先使用官方文档中的示例格式

##### **代码示例**
- **问题代码**：
```python
# 使用OpenAI格式的工具类型
tool = {
    "type": "function",
    "function": func
}
```

- **修复代码**：
```python
# 转换为Claude支持的custom格式
tool = {
    "type": "custom",
    "name": func.get("name", "未命名工具"),
    "description": func.get("description", ""),
    "tool_schema": func.get("parameters", {})
}
```

- **关键差异**：
  1. 将工具类型从`function`改为`custom`
  2. 将OpenAI的function参数结构转换为Claude需要的顶级参数
  3. 确保工具名称、描述和参数架构正确传递
  4. 添加工具格式验证和错误处理

#### **2025-03-03 17:05:00 - [API-104] Claude API工具选择参数格式错误 [P1]**
- **标签**：#工具调用 #API格式 #Claude接口
- **参考**：TOOL-001, API-103

##### **错误信息**
- **现象**：Claude API请求返回400错误，提示tool_choice参数格式不正确
- **错误日志**：
  ```
  HTTP 400: {"type":"error","error":{"type":"invalid_request_error","message":"tool_choice: Input should be a valid dictionary or object to extract fields from"}}
  ```
- **影响范围**：所有使用工具调用功能的接口
- **复现条件**：使用OpenAI格式发送包含工具的请求到Claude API

##### **错误分析**
- **根本原因**：Claude API的tool_choice参数需要是一个字典格式（如`{"type": "auto"}`），但代码中传递的是字符串格式（如`"auto"`）
- **尝试的方案**：
  1. 检查Claude和OpenAI API文档对比tool_choice参数格式差异：
     - 结果：发现不匹配
     - 原因：Claude要求以字典形式提供tool_choice参数，而不是字符串
  2. 修改tool_choice参数处理逻辑：
     - 结果：成功
     - 验证：API调用成功，不再返回400错误

##### **解决方案**
- **最终方案**：
  1. 在claude_client.py中更新tool_choice处理逻辑，将字符串转换为字典格式
  ```python
  # 将字符串转换为字典格式
  if isinstance(tool_choice, str):
      if tool_choice in ["auto", "none"]:
          tool_choice = {"type": tool_choice}
      else:
          tool_choice = {"type": "auto"}
  ```
  2. 修改main.py和deepclaude.py中相关参数类型和日志处理
  3. 对工具选择策略添加更详细的日志记录
- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：低
  - 资源消耗：低
  - 稳定性：高
  - 兼容性：高

##### **经验总结**
- **避免方向**：
  1. 避免直接假设API参数格式兼容，尤其是在处理不同平台API时
  2. 避免在参数类型上过于严格限制（如使用`tool_choice: str = "auto"`）
  3. 避免缺少对API参数格式的详细日志记录

- **正确思路**：
  1. 查阅最新API文档，确认参数的精确格式要求
  2. 实现灵活的参数格式转换
  3. 在转换前后添加详细日志
  4. 使用更灵活的参数类型定义（移除不必要的类型限制）

- **最佳实践**：
  1. 记录API请求的完整参数和格式
  2. 在格式转换前后记录参数变化
  3. 灵活处理不同API之间的参数格式差异
  4. 优先使用官方文档中的示例格式

##### **代码示例**
- **问题代码**：
```python
# 直接将字符串格式的tool_choice传给Claude API
data["tool_choice"] = "auto"
```

- **修复代码**：
```python
# 将字符串转换为字典格式
if isinstance(tool_choice, str):
    tool_choice = {"type": tool_choice}
data["tool_choice"] = tool_choice
```

- **关键差异**：
  1. 将字符串格式的tool_choice转换为Claude API要求的字典格式
  2. 添加类型检查，灵活处理不同格式的参数
  3. 增加详细日志，记录参数转换过程
  4. 移除不必要的参数类型限制 

#### **2025-03-10 11:30:00 - [CODE-003] chat_completions_with_stream方法中的代码重复问题 [P3]**
- **标签**：#代码优化 #代码重复 #流处理
- **参考**：TOOL-001, API-105

##### **错误信息**
- **现象**：`chat_completions_with_stream`方法中存在重复验证和处理工具的代码，造成逻辑冗余
- **影响范围**：工具调用流式处理功能
- **复现条件**：使用工具调用功能时

##### **错误分析**
- **根本原因**：
  1. 代码修改过程中产生了重复的工具验证和处理逻辑
  2. 条件判断嵌套过深，导致相似代码被重复编写
  3. 流程分支过多，处理逻辑不统一

- **尝试的方案**：
  1. 重构方法，统一工具处理逻辑：
     - 结果：成功
     - 验证：代码更简洁，逻辑更清晰

##### **解决方案**
- **最终方案**：
  1. 移除重复的工具验证代码
  2. 统一参数准备逻辑，使用基础参数字典
  3. 根据工具有效性决定是否添加工具相关参数
  4. 使用统一的API调用入口

- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：低
  - 资源消耗：不变
  - 稳定性：提高
  - 可维护性：显著提高

##### **经验总结**
- **避免方向**：
  1. 避免复制粘贴代码而不检查重复
  2. 避免在条件分支中编写相似逻辑
  3. 避免在修复问题时引入新的代码冗余

- **正确思路**：
  1. 提取共用逻辑到单一位置
  2. 使用统一的参数准备和校验方式
  3. 明确分支条件和处理逻辑
  4. 定期检查和重构冗余代码

- **最佳实践**：
  1. 维护单一职责原则
  2. 使用参数字典来统一管理API调用参数
  3. 提取重复验证逻辑为独立函数
  4. 代码审查时关注重复逻辑

##### **预防措施**
1. 定期代码审查关注重复模式
2. 修改复杂方法前先整体理解逻辑
3. 引入单元测试验证代码行为不变
4. 维护更新代码注释解释复杂逻辑 

#### **2025-03-10 14:20:00 - [API-106] Claude API工具自定义字段嵌套错误 [P1]**
- **标签**：#工具调用 #API格式 #Claude接口
- **参考**：API-105, TOOL-001

##### **错误信息**
- **现象**：Claude API请求返回400错误，提示custom类型工具中不允许有额外的type字段
- **错误日志**：
  ```
  HTTP 400: {"type":"error","error":{"type":"invalid_request_error","message":"tools.0.custom.type: Extra inputs are not permitted"}}
  ```
- **影响范围**：使用自定义工具的所有接口
- **复现条件**：使用带有嵌套type字段的自定义工具

##### **错误分析**
- **根本原因**：Claude API对于自定义工具(custom类型)的格式要求严格，不允许在custom对象内再有嵌套的type字段。但在格式转换过程中，可能出现了将type字段错误地保留或复制到custom对象内部的情况。
- **尝试的方案**：
  1. 检查全部工具转换和验证代码：
     - 结果：发现多处可能导致嵌套type字段的地方
     - 原因：各层进行工具转换和验证时未清理嵌套字段
  2. 修改所有相关代码部分：
     - 结果：成功
     - 验证：API调用不再返回400错误

##### **解决方案**
- **最终方案**：
  1. 在三处关键位置添加对自定义工具嵌套type字段的检查和清理：
     - main.py中的工具转换代码
     - deepclaude.py的_validate_and_convert_tools方法
     - claude_client.py的_prepare_request_data方法
  2. 添加详细日志，记录发现和修复嵌套type字段的情况
  3. 确保在所有工具处理环节都检查格式合规性
- **评估**：
  - 有效性：完全解决（100%）
  - 复杂度：低
  - 资源消耗：不变
  - 稳定性：高
  - 兼容性：高

##### **经验总结**
- **避免方向**：
  1. 避免假设API的格式要求与文档说明完全一致
  2. 避免在多层转换过程中保留可能有问题的字段
  3. 避免仅依靠一处验证来确保数据格式正确

- **正确思路**：
  1. 完整理解API的格式要求，注意隐含的限制条件
  2. 在数据发送前进行最后的格式检查
  3. 添加多层验证确保数据格式符合要求
  4. 记录详细日志用于调试和问题追踪

- **最佳实践**：
  1. 在每个处理工具的关键节点都添加格式验证
  2. 使用深拷贝而非浅拷贝修改对象，避免意外修改原始数据
  3. 严格按照API文档示例构造数据
  4. 对复杂嵌套结构的数据进行详细的日志记录

##### **代码示例**
- **问题代码**：
```python
# 转换为Claude支持的custom格式，但保留了嵌套的type字段
claude_tool = {
    "type": "custom",
    "custom": {
        "type": "function",  # 这是多余的嵌套字段
        "name": func.get("name"),
        "description": func.get("description"),
        "tool_schema": func.get("parameters", {})
    }
}
```

- **修复代码**：
```python
# 正确的Claude自定义工具格式
claude_tool = {
    "type": "custom",
    "name": func.get("name"),
    "description": func.get("description"),
    "tool_schema": func.get("parameters", {})
}

# 或者检查并修复嵌套字段
if "custom" in tool and isinstance(tool["custom"], dict) and "type" in tool["custom"]:
    tool["custom"].pop("type", None)
```

- **关键差异**：
  1. 移除了自定义工具中不必要的嵌套type字段
  2. 添加了对嵌套字段的检查和清理
  3. 遵循Claude API对custom工具的格式要求
  4. 增加了详细的日志和错误处理