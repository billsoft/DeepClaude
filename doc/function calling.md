# OpenAI API 函数调用响应格式详解

根据您提供的请求样例，我将详细解析OpenAI API在函数调用下的流式响应(stream=true)格式。

## 基本响应结构

当OpenAI API接收到函数调用请求时，使用stream=true参数时，响应会以多个数据块(chunks)形式返回，每个数据块是一个JSON对象，以`data: `前缀开始。[1]

响应的基本结构如下：
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "choices":[{"index":0,"delta":{"role":"assistant", "content":null, "function_call": {"name": "ddgo_search", "arguments": "{"}},"finish_reason":null}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "choices":[{"index":0,"delta":{"function_call": {"arguments": "\"query\": \""}},"finish_reason":null}]}
...
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "choices":[{"index":0,"delta":{"function_call": {"arguments": "\"}"}},"finish_reason":null}]}
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}
data: [DONE]
```
[2]

## 函数调用流式响应的组成部分

1. **初始块**：包含函数调用的基本信息，设置role为assistant并开始function_call
2. **参数块**：一系列包含函数参数的块，通常分段发送JSON字符串
3. **结束块**：包含finish_reason为"function_call"的块
4. **完成标记**：最后发送`data: [DONE]`表示流结束 
[1], [4]

## 重要字段解释

对于您的请求案例，响应会包含以下关键部分：

- **id**: 响应的唯一标识符
- **object**: 始终为"chat.completion.chunk"(流式响应)
- **created**: UNIX时间戳，表示响应创建时间
- **model**: 使用的模型名称，您请求中是"gpt-4"
- **choices**: 包含模型生成的实际内容
  - **delta**: 表示当前块相对于前一块的差异部分
    - **function_call**: 包含函数调用信息
      - **name**: 函数名称，您示例中为"ddgo_search"
      - **arguments**: 函数参数的JSON字符串，会分多块发送
  - **finish_reason**: 生成结束的原因，函数调用完成时为"function_call"
[3]

## 完整流程示例

基于您提供的例子，完整响应过程应该是：

1. 第一个块：初始化函数调用
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{"role":"assistant", "content":null, "function_call": {"name": "ddgo_search", "arguments": "{"}},"finish_reason":null}]}
```

2. 参数块：逐步发送查询参数(分多块发送)
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{"function_call": {"arguments": "\"query\": \""}},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{"function_call": {"arguments": "沈阳天气"}},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{"function_call": {"arguments": "\""}},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{"function_call": {"arguments": "}"}},"finish_reason":null}]}
```

3. 结束块：表示函数调用完成
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4", "choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}
```

4. 流结束标记
```
data: [DONE]
```
[2], [4]

## 如果包含usage信息

由于您的请求中指定了`"stream_options": {"include_usage": true}`，在最后的`data: [DONE]`之前，API还会发送一个包含tokens使用统计的块：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1709520478,"model":"gpt-4","choices":[],"usage":{"prompt_tokens":128,"completion_tokens":42,"total_tokens":170}}
```
[3]

## 客户端处理流式响应

客户端需要:
1. 解析每个数据块的JSON内容
2. 逐步构建完整的函数调用参数
3. 当接收到finish_reason为"function_call"的块时，执行相应函数
4. 处理后续的模型响应
[1], [4]

OpenAI官方SDK已经处理了这些复杂性，但如果您是自己实现客户端，需要正确处理这些数据块拼接。[3]

---

[1]: [OpenAI Function Calling Official Documentation](https://platform.openai.com/docs/guides/function-calling)
[2]: [OpenAI API Reference - Streaming](https://platform.openai.com/docs/api-reference/streaming)
[3]: [OpenAI Community - Function Calling Response Format](https://community.openai.com/t/function-calling-response-format/920969)
[4]: [Stack Overflow - Return arguments from function calling with OpenAI API when streaming](https://stackoverflow.com/questions/77728888/return-arguments-from-function-calling-with-openai-api-when-streaming)

# OpenAI模型工具调用的完整交互流程分析

根据提供的日志，我将详细分析OpenAI API如何处理工具调用(Function Calling)请求，从接收请求到生成回复的全过程。

## 一、初始请求处理阶段

从日志可见，服务器接收到了一个标准的OpenAI API请求，包含以下关键要素:

1. **请求内容**: 
   - 系统指令要求"始终使用中文沟通"并"在回答复杂问题时要先调用搜索工具tavily"
   - 用户查询简单明了:"沈阳天气"
   - 使用的模型是"gpt-4o"
   - 启用了流式响应(stream=true)和使用统计(include_usage=true)[1]

2. **工具定义**:
   ```json
   "functions": [
     {
       "name": "tavily_search",
       "description": "A tool for search engine built specifically for AI agents...",
       "parameters": {"properties": {"query": {"description": "The search query.", "type": "string"}}, "required": ["query"]}
     },
     {
       "name": "tavily_extract", 
       "description": "A tool for extracting raw content from web pages...",
       "parameters": {"properties": {"urls": {"description": "A comma-separated list of URLs...", "type": "string"}}, "required": ["urls"]}
     }
   ]
   ```

3. **请求转换**: 
   日志中显示:"检测到 OpenAI 格式的 functions 定义，正在转换为 tools 格式" - 这表明服务器在内部将旧版的functions格式转换为新版tools格式[2]

## 二、模型决策与工具调用流程

接收到请求后，OpenAI模型会执行以下步骤:

1. **模型评估与决策**: 
   - 模型分析用户的"沈阳天气"查询
   - 根据系统指令("在回答有实时性要求的问题时要先调用工具中的搜索功能")，模型决定这需要最新信息[1]
   - 模型选择调用tavily_search工具获取实时天气信息

2. **生成工具调用的流式响应**:
   由于stream=true，模型会分块返回工具调用的JSON格式:
   ```
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"tavily_search","arguments":"{"}}]},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"query\": \""}}]},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"沈阳天气"}}]},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\""}}]},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
   ```
   [3]

## 三、客户端处理与工具执行

收到上述流式响应后，客户端需要进行以下处理:

1. **解析工具调用数据**:
   - 将各个数据块中的arguments部分拼接起来
   - 最终获得完整的工具调用信息:
     ```json
     {
       "name": "tavily_search",
       "arguments": {
         "query": "沈阳天气"
       }
     }
     ```
   [4]

2. **执行工具调用**:
   - 客户端需要实际调用tavily_search工具
   - 向Tavily API发送"沈阳天气"作为查询参数
   - 获取搜索结果(例如包含沈阳当前温度、天气状况、未来预报等信息)[3]

## 四、将工具结果返回模型

执行完工具调用后，客户端需要:

1. **构建后续请求**:
   - 将原始的消息历史和工具调用结果一起发送给OpenAI API
   ```json
   {
     "messages": [
       {"role": "system", "content": "始终使用中文沟通..."},
       {"role": "user", "content": "沈阳天气"},
       {"role": "assistant", "tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "tavily_search", "arguments": "{\"query\": \"沈阳天气\"}"}}]},
       {"role": "tool", "tool_call_id": "call_xxx", "content": "{搜索引擎返回的沈阳天气数据}"}
     ],
     "model": "gpt-4o",
     "stream": true
     ...
   }
   ```
   [1], [4]

2. **模型生成最终回复**:
   - OpenAI接收到包含工具调用结果的请求
   - 模型基于工具返回的信息生成自然语言回复
   - 回复以流式方式返回给客户端:
   ```
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"根据"},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"最新"},"finish_reason":null}]}
   ...
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"。"},"finish_reason":null}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1646849159,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":239,"completion_tokens":127,"total_tokens":366}}
   data: [DONE]
   ```
   [3], [5]

## 五、完整流程总结

整个工具调用流程形成一个闭环:

1. 用户发送"沈阳天气"的查询
2. 模型决定需要调用工具，生成工具调用指令
3. 客户端解析工具调用，执行tavily_search
4. 搜索获取沈阳实时天气信息
5. 客户端将工具调用结果返回给模型
6. 模型基于最新数据生成人类可读的天气报告
7. 客户端将最终回复呈现给用户[1], [5]

整个过程中，API维护了完整的对话上下文，并通过流式传输减少了响应延迟，让体验更加流畅。[4]

---

[1]: [OpenAI API Reference - Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
[2]: [OpenAI Function Calling - Migration Guide](https://platform.openai.com/docs/guides/function-calling/migration-guide)
[3]: [OpenAI Cookbook - Function Calling with Streaming](https://cookbook.openai.com/examples/how_to_use_function_calling_with_streaming)
[4]: [OpenAI API - Tool Calls Response Format](https://platform.openai.com/docs/api-reference/chat/object#chat/object-tool_calls)
[5]: [OpenAI Community Discussion - Tool Use with Multiple Steps](https://community.openai.com/t/tool-use-with-multiple-steps/1122303)


# Claude 3.7 工具调用的完整交互流程分析

根据提供的日志，我将详细分析Claude 3.7如何处理工具调用(Function Calling)请求，从接收请求到生成回复的完整流程，特别是在接收与OpenAI兼容的API格式时。

## 一、初始请求接收与格式转换

首先可以看到，请求是使用OpenAI客户端库(1.64.0)发送的，但请求目标是Claude 3.7模型：

1. **请求头信息**:
   - 使用OpenAI Python客户端(version 1.64.0)
   - 运行环境：Linux arm64，Python 3.12.3
   - Authorization头：`Bearer DeepSYSAI`[1]

2. **请求体关键内容**:
   ```json
   {
     "messages": [...],
     "model": "gpt-4o",  // 注意：实际将由系统映射到Claude 3.7
     "functions": [...],  // OpenAI格式的函数定义
     "stream": true,
     "stream_options": {"include_usage": true}
   }
   ```
   
3. **格式转换处理**:
   - 日志显示："检测到 OpenAI 格式的 functions 定义，正在转换为 tools 格式"
   - 系统将OpenAI的`functions`格式转换为Claude API所需的`tools`格式[2]
   - 这种转换是必需的，因为Anthropic和OpenAI的工具调用格式有所不同[3]

## 二、Claude 3.7的请求理解与决策过程

当Claude 3.7收到转换后的请求后，会执行以下处理：

1. **系统指令解析**:
   - 解析"始终使用中文沟通"指令
   - 理解"在回答有实时性要求的问题时要调用搜索工具tavily"的指令(权重500%)[1]
   
2. **查询评估与工具选择**:
   - Claude分析"沈阳天气"查询，识别为需要实时信息的查询
   - 评估可用工具列表(tavily_search和tavily_extract)
   - 决定使用`tavily_search`工具，因为它适合获取实时信息[4]
   - Claude不会使用`tavily_extract`，因为此时还没有需要提取内容的URL

## 三、Claude特有的工具调用流式响应格式

Claude 3.7生成的工具调用流式响应与OpenAI有显著不同：

1. **开始标记块**:
   ```
   data: {"type":"message_start","message":{"id":"msg_01Xabc","type":"message","role":"assistant","content":[],"model":"claude-3-7-sonnet-20240229","stop_reason":null,"stop_sequence":null}}
   ```

2. **工具调用块**:
   ```
   data: {"type":"content_block_start","content_block":{"id":"cbid_abc","type":"tool_use","index":0}}
   data: {"type":"tool_use_text","tool_use":{"name":"tavily_search","input":{"query":"沈阳天气"},"id":"tu_01abc"}}
   data: {"type":"content_block_stop","content_block":{"id":"cbid_abc","type":"tool_use","index":0}}
   ```

3. **消息完成块**:
   ```
   data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null}}
   data: {"type":"message_stop"}
   ```
   [5], [3]

注：由于原请求是OpenAI格式，系统会在后台将Claude响应格式转换为与OpenAI兼容的格式返回给客户端。[2]

## 四、客户端处理与工具执行

客户端接收到工具调用指令后的处理流程：

1. **响应解析**:
   - 客户端接收兼容OpenAI格式的流式响应
   - 解析得到完整工具调用参数：`{"query": "沈阳天气"}`
   - 识别需要调用的工具名称：`tavily_search`[6]

2. **工具执行过程**:
   - 客户端调用Tavily Search API
   - 发送请求参数：`{"query": "沈阳天气"}`
   - Tavily API执行网络搜索
   - 返回搜索结果，包含沈阳当前天气状况、温度、预报等信息[4]

## 五、工具结果提交回Claude 3.7

执行完工具调用后，客户端需要将结果提交回Claude 3.7：

1. **构建后续请求**:
   ```json
   {
     "messages": [
       {"role": "system", "content": "始终使用中文沟通..."},
       {"role": "user", "content": "沈阳天气"},
       {"role": "assistant", "content": null, "tool_calls": [{"id": "tu_01abc", "type": "function", "function": {"name": "tavily_search", "arguments": "{\"query\": \"沈阳天气\"}"}}]},
       {"role": "tool", "tool_call_id": "tu_01abc", "content": "{天气搜索结果JSON}"}
     ],
     "model": "claude-3-7-sonnet-20240229",
     "stream": true
   }
   ```
   [3], [7]

2. **Claude的工具结果处理**:
   - Claude接收包含工具结果的新请求
   - 解析tavily_search返回的天气信息
   - 提取关键天气数据(温度、湿度、风向、预报等)[7]

## 六、Claude基于工具结果生成最终回复

Claude 3.7接收到工具执行结果后，会生成最终回复：

1. **回复生成流程**:
   - 分析搜索结果中的沈阳天气信息
   - 按照系统提示"始终使用中文"，以中文组织回答
   - 遵循"细致详细解答(1000%权重)"指令，提供全面的天气信息
   - 生成包含当前温度、天气状况、未来预报等的完整回复[5]

2. **流式回复格式**:
   ```
   data: {"type":"message_start","message":{"id":"msg_02Xdef","role":"assistant","content":[],"model":"claude-3-7-sonnet-20240229"}}
   data: {"type":"content_block_start","content_block":{"id":"cbid_def","type":"text","index":0}}
   data: {"type":"text","text":"根据最新搜索结果，沈阳今天"}
   data: {"type":"text","text":"的天气情况如下：\n\n当前温度："}
   ...
   data: {"type":"content_block_stop","content_block":{"id":"cbid_def","type":"text","index":0}}
   data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}
   data: {"type":"message_stop"}
   ```
   
   这些响应也会被系统转换为OpenAI兼容格式后返回给客户端。[3], [5]

## 七、Claude与OpenAI工具调用的关键差异

尽管系统进行了格式转换，Claude和OpenAI处理工具调用有一些本质区别：

1. **工具调用格式**:
   - Claude使用`tool_use`内容块，OpenAI使用`function`对象
   - Claude的工具调用有独立ID和完整JSON格式参数
   - OpenAI以增量方式流式传输函数参数字符串[2], [3]

2. **流式响应结构**:
   - Claude使用更明确的事件类型标记(`message_start`, `content_block_start`等)
   - OpenAI使用delta更新机制，逐步构建完整响应[5], [8]

3. **多工具调用处理**:
   - Claude支持在单个响应中调用多个工具，每个工具调用有单独的内容块
   - OpenAI使用数组索引区分多个工具调用[7]

## 八、完整工作流总结

整个Claude 3.7工具调用流程形成闭环：

1. 接收OpenAI格式请求并内部转换为Claude格式
2. 分析查询，决定调用tavily_search工具
3. 生成工具调用指令并以兼容格式返回
4. 客户端执行工具调用获取沈阳天气信息
5. 将工具结果提交回Claude
6. Claude分析工具结果并生成完整天气报告
7. 将回复以兼容格式流式返回给客户端[4], [7]

这种设计不仅保持了与OpenAI API的兼容性，同时充分发挥了Claude 3.7在工具使用方面的能力，实现了高效、准确的信息检索和回复生成。[3], [8]

---

[1]: [Anthropic API Reference - Messages](https://docs.anthropic.com/claude/reference/messages_post)
[2]: [Anthropic Tool Use Tutorial - OpenAI Compatibility](https://docs.anthropic.com/claude/docs/tool-use-with-claude#openai-compatibility)
[3]: [Anthropic vs OpenAI Tool Formats](https://community.anthropic.com/t/tool-use-with-claude-api/3419)
[4]: [Claude's Tool Use Capabilities](https://docs.anthropic.com/claude/docs/tool-use-with-claude)
[5]: [Anthropic Streaming Response Format](https://docs.anthropic.com/claude/reference/messages-streaming)
[6]: [Tool Use Client Implementation](https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use)
[7]: [Anthropic Multi-tool Usage Examples](https://docs.anthropic.com/claude/docs/tool-use-examples)
[8]: [Claude API Best Practices - Tool Use](https://docs.anthropic.com/claude/docs/api-best-practices)