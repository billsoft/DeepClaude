# Web Function Calling 完整工作流解析

## 总体工作流程

1. **初始请求** → **工具调用决策** → **工具执行** → **结果获取** → **最终回答**

这个流程在DeepClaude中需要多次与Claude模型交互，并涉及消息组织、转换和拼接。

## 详细交互流程

### 1. 初始请求阶段 (第一次交互)

**客户端请求格式**（OpenAI格式）:
```json
{
    "messages": [
        {"role": "system", "content": "系统指令..."},
        {"role": "user", "content": "沈阳天气"}
    ],
    "functions": [...工具定义...],
    "stream": true
}
```

**转换过程**:
```python
# main.py 中的转换逻辑
tools = []
for func in functions:
    tool = {
        "name": func.get("name"),
        "description": func.get("description"),
        "input_schema": {...}  # 转换函数参数
    }
    tools.append(tool)
```

**发送给Claude的请求体**:
```json
{
    "model": "claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "沈阳天气"}],
    "stream": true,
    "system": "系统指令...",
    "tools": [...转换后的工具定义...],
    "tool_choice": {"type": "auto"}
}
```

### 2. 工具调用决策阶段

**Claude返回事件流**:
- `message_start`: 消息开始
- `content_block_start`: 内容块开始（文本）
- `content_block_delta`: 文本内容（"我需要使用搜索工具..."）
- `content_block_stop`: 内容块结束
- `content_block_start`: 工具使用块开始（type=tool_use）
- `content_block_delta`: 多个input_json_delta事件（工具参数片段）
- `content_block_stop`: 工具使用块结束
- `message_delta`: 消息状态更新
- `message_stop`: 消息结束

**工具调用提取**:
```python
# 从事件流中提取工具调用
tool_call = {
    "id": str(uuid.uuid4()),
    "type": "function",
    "function": {
        "name": tool_name,  # 如"tavily_search"
        "arguments": json.dumps({"query": "沈阳天气"})
    }
}
```

### 3. 工具执行阶段

**执行工具**:
```python
# core.py 中的工具执行逻辑
result = await self._execute_tool_call({
    "tool": tool_name,
    "tool_input": tool_args
})
tool_results.append({
    "role": "user",
    "name": tool_name,
    "content": result
})
```

### 4. 结果拼接与最终回答阶段 (第二次交互)

**关键步骤**：拼接新的消息数组，包含原始问题、工具结果和工具执行结果

```python
# 拼接新消息
assistant_message = {
    "role": "assistant",
    "content": "我需要查询信息以回答您的问题"
}

tool_results_content = ""
for result in tool_results:
    tool_name = result.get("name")
    tool_content = result.get("content")
    tool_results_content += f"### {tool_name}工具的执行结果 ###\n{tool_content}\n\n"

user_message = {
    "role": "user",
    "content": tool_results_content
}

# 创建新的消息数组
new_messages = copy.deepcopy(messages)  # 保留原始消息
new_messages.append(assistant_message)  # 添加助手消息
new_messages.append(user_message)       # 添加工具结果消息
```

**发送给Claude生成最终回答**:
```json
{
    "model": "claude-3-7-sonnet-20250219",
    "messages": [
        {"role": "user", "content": "沈阳天气"},
        {"role": "assistant", "content": "我需要查询信息以回答您的问题"},
        {"role": "user", "content": "### tavily_search工具的执行结果 ###\n沈阳今日天气晴朗，气温20-25度...\n\n"}
    ],
    "temperature": 0.7,
    "top_p": 0.9
}
```

## 核心问题及修复

当前系统失败的关键点是**工具调用事件流的处理**。Claude发送了工具调用事件，但系统没有正确解析和处理这些事件:

1. **事件流解析错误**：
   - 使用了不兼容的`iter_lines()`方法
   - 需修改为`iter_any()`并手动处理事件边界

2. **工具调用状态跟踪缺失**：
   - 没有正确跟踪和累积工具调用数据
   - 需添加状态变量记录当前工具名称、ID和参数累积状态

3. **JSON增量更新处理缺失**：
   - `input_json_delta`事件需要特殊处理来构建完整参数
   - 需添加递归JSON更新方法合并增量更新

## 修复后的流程

修复后，当系统收到"沈阳天气"这样的查询时:

1. Claude识别这是需要实时信息的查询
2. Claude通过事件流返回决定使用`tavily_search`工具
3. 系统提取工具调用，执行工具，获取结果（天气信息）
4. 系统将原始问题和工具结果重新发送给Claude
5. Claude基于工具结果生成最终回答，提供完整的天气信息

这样，每个查询实际上涉及**至少两轮**与Claude的交互，才能完成完整的工具调用与回答流程。