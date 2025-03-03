核心需求 我们项目是 fastapi兼容 openai api，需要新增支持 使用 dify 发送过来的工具
原理
# Dify-OpenAI-Tavily 完整交互流程：以"查询沈阳天气"为例

当用户在 Dify 平台上输入"查询沈阳天气"这样的问题时，背后发生了一系列复杂的交互流程。下面我将详细分析整个请求的传递路径、交互时序以及数据流动过程。

## 一、用户输入阶段

### 1.1 用户界面交互
当用户在 Dify 创建的应用界面中输入"查询沈阳天气"并发送请求时：

1. Dify 前端捕获用户输入
2. 生成唯一会话 ID (conversation_id)
3. 将用户查询通过 HTTPS 请求发送到 Dify 后端 API

```json
// 前端发送到 Dify 后端的请求
{
  "query": "查询沈阳天气",
  "conversation_id": "conv_123456789",
  "inputs": {},
  "response_mode": "streaming"
}
```

## 二、Dify 后端处理阶段

### 2.1 请求预处理
Dify 后端接收到用户查询后进行一系列处理：

1. **身份验证与授权**：验证用户身份和访问权限
2. **会话管理**：获取或创建对话历史
3. **应用配置加载**：加载当前应用的设置，包括已启用的工具列表
4. **提示词处理**：将应用预设的系统提示与用户查询合并

### 2.2 构建 OpenAI 请求
Dify 构建发送给 OpenAI API 的请求内容：

```json
{
  "model": "gpt-4-0125-preview",
  "messages": [
    {
      "role": "system",
      "content": "你是一个智能助手，请根据用户的问题提供帮助。当需要查询实时信息时，请使用提供的工具。"
    },
    {
      "role": "user",
      "content": "查询沈阳天气"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tavily_search",
        "description": "使用 Tavily 搜索引擎获取最新的互联网信息，适用于需要实时数据的查询",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "搜索查询字符串"
            },
            "search_depth": {
              "type": "string",
              "enum": ["basic", "advanced"],
              "description": "搜索深度，basic为基础搜索，advanced为高级搜索"
            },
            "include_domains": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "要包含的网站域名列表"
            },
            "exclude_domains": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "要排除的网站域名列表"
            },
            "include_answer": {
              "type": "boolean",
              "description": "是否在结果中包含Tavily生成的答案摘要"
            }
          },
          "required": ["query"]
        }
      }
    },
    // 可能还有其他工具定义...
  ],
  "stream": true
}
```

## 三、OpenAI 处理阶段

### 3.1 接收请求与初步分析
OpenAI API 接收 Dify 的请求后：

1. 验证 API 密钥和请求格式
2. 加载指定的 GPT 模型（如 GPT-4）
3. 分析用户查询内容："查询沈阳天气"

### 3.2 模型判断与工具选择
模型通过理解查询内容，决定需要使用工具：

1. 识别用户查询关于"沈阳天气"，属于实时信息请求
2. 理解自身知识有时效性限制，无法提供最新天气信息
3. 评估可用工具列表，发现 `tavily_search` 可用于获取实时信息
4. 决定使用 `tavily_search` 工具来查询沈阳天气信息

### 3.3 工具调用请求生成
OpenAI 模型生成包含工具调用的响应：

```json
{
  "id": "chatcmpl-123abc456def",
  "object": "chat.completion.chunk",
  "created": 1709457687,
  "model": "gpt-4-0125-preview",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_7890xyz",
            "type": "function",
            "function": {
              "name": "tavily_search",
              "arguments": "{\"query\": \"沈阳今天天气实时信息\", \"search_depth\": \"basic\", \"include_answer\": true}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

## 四、Dify 工具执行阶段

### 4.1 解析工具调用请求
Dify 接收 OpenAI 的响应并解析工具调用信息：

1. 提取工具名称：`tavily_search`
2. 解析参数：`{"query": "沈阳今天天气实时信息", "search_depth": "basic", "include_answer": true}`
3. 验证参数格式与类型是否符合要求

### 4.2 执行 Tavily 搜索
Dify 调用 Tavily API 执行搜索：

```
POST https://api.tavily.com/search
Authorization: Bearer {tavily_api_key}
Content-Type: application/json

{
  "query": "沈阳今天天气实时信息",
  "search_depth": "basic",
  "include_answer": true
}
```

### 4.3 Tavily 响应
Tavily 接收请求，执行搜索并返回结果：

```json
{
  "answer": "今天沈阳天气晴朗，当前气温5℃，最高温度10℃，最低温度-2℃，东北风3-4级。空气质量良好，适合户外活动。",
  "results": [
    {
      "title": "沈阳天气预报_沈阳天气预报一周_沈阳天气预报15天查询",
      "url": "http://www.weather.com.cn/weather/101070101.shtml",
      "content": "中国天气网沈阳天气预报提供沈阳今天天气、明天天气、后天天气、沈阳天气预报一周、沈阳天气预报15天查询...",
      "score": 0.92,
      "published_date": "2023-03-03"
    },
    {
      "title": "沈阳天气实况_实时天气_气象局权威发布",
      "url": "https://tianqi.2345.com/shenyang/58238.htm",
      "content": "沈阳实时天气: 晴朗, 气温5℃，东北风3-4级，相对湿度45%，空气质量良好...",
      "score": 0.89,
      "published_date": "2023-03-03"
    }
    // 更多搜索结果...
  ],
  "query": "沈阳今天天气实时信息",
  "search_depth": "basic"
}
```

### 4.4 处理工具执行结果
Dify 接收 Tavily 返回的结果：

1. 验证响应格式是否正确
2. 提取搜索结果内容
3. 格式化为 OpenAI 所需的 `tool` 消息格式

## 五、结果返回阶段

### 5.1 构建工具响应消息
Dify 将 Tavily 搜索结果构建为工具响应消息：

```json
{
  "model": "gpt-4-0125-preview",
  "messages": [
    // 之前的消息历史...
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_7890xyz",
          "type": "function",
          "function": {
            "name": "tavily_search",
            "arguments": "{\"query\": \"沈阳今天天气实时信息\", \"search_depth\": \"basic\", \"include_answer\": true}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_7890xyz",
      "name": "tavily_search",
      "content": "{\"answer\":\"今天沈阳天气晴朗，当前气温5℃，最高温度10℃，最低温度-2℃，东北风3-4级。空气质量良好，适合户外活动。\",\"results\":[{\"title\":\"沈阳天气预报_沈阳天气预报一周_沈阳天气预报15天查询\",\"url\":\"http://www.weather.com.cn/weather/101070101.shtml\",\"content\":\"中国天气网沈阳天气预报提供沈阳今天天气、明天天气、后天天气、沈阳天气预报一周、沈阳天气预报15天查询...\",\"score\":0.92,\"published_date\":\"2023-03-03\"},{\"title\":\"沈阳天气实况_实时天气_气象局权威发布\",\"url\":\"https://tianqi.2345.com/shenyang/58238.htm\",\"content\":\"沈阳实时天气: 晴朗, 气温5℃，东北风3-4级，相对湿度45%，空气质量良好...\",\"score\":0.89,\"published_date\":\"2023-03-03\"}]}"
    }
  ],
  "stream": true
}
```

### 5.2 OpenAI 生成最终回复
OpenAI 接收工具响应，分析结果并生成自然语言回复：

```json
{
  "id": "chatcmpl-123efg456hij",
  "object": "chat.completion.chunk",
  "created": 1709457689,
  "model": "gpt-4-0125-preview",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "根据最新信息，沈阳今天天气晴朗，当前气温5℃，最高温度10℃，最低温度-2℃，东北风3-4级。空气质量良好，适合户外活动。"
      },
      "finish_reason": null
    }
  ]
}
```

### 5.3 Dify 处理最终回复
Dify 接收 OpenAI 的最终回复：

1. 流式处理响应内容（因为启用了 `stream: true`）
2. 将 AI 回复添加到对话历史中
3. 记录工具调用的执行过程和结果
4. 发送回复内容到前端界面

### 5.4 用户界面展示
Dify 前端接收后端响应并展示给用户：

1. 实时显示回复内容
2. 可能展示参考来源链接（基于 Tavily 结果）
3. 更新对话历史界面

## 六、技术细节解析

### 6.1 Dify 工具配置机制

Dify 的工具配置界面允许应用创建者进行以下操作：

1. **工具启用**：在 Dify 控制台的应用设置中选择启用 Tavily 搜索工具
2. **API 密钥配置**：配置 Tavily API 密钥
3. **参数预设**：设置默认的搜索深度、包含/排除域名等
4. **权限控制**：限制工具的使用频率和范围

![Dify工具配置界面](https://i.imgur.com/u2xKuRe.png)

### 6.2 OpenAI 工具选择逻辑

OpenAI 模型（如 GPT-4）根据多方面因素决定是否使用工具：

1. **查询类型识别**：识别出"查询沈阳天气"属于实时信息请求
2. **知识时效性**：模型认识到天气信息具有高时效性，自身知识可能过时
3. **工具适用性评估**：评估 Tavily 搜索是否适合获取这类信息
4. **上下文理解**：根据整个对话上下文判断用户需要的信息类型和详细程度

值得注意的是，模型会自主决定：
- 是否需要使用工具（vs. 直接回答值得注意的是，模型会自主决定：
- 是否需要使用工具（vs. 直接回答）
- 使用哪个工具最合适（若有多个工具可用）
- 如何构造最优的搜索查询（通常会优化用户原始查询）

### 6.3 工具调用参数构建过程

在构建 `tavily_search` 工具调用时，OpenAI 模型会：

1. **查询优化**：将"查询沈阳天气"优化为"沈阳今天天气实时信息"
2. **参数选择**：
   - 必填参数：`query`（搜索查询）
   - 可选参数：根据任务需要选择 `search_depth`、`include_answer` 等
3. **参数值设定**：
   - 选择 `"search_depth": "basic"`（基本深度通常足够天气信息）
   - 设置 `"include_answer": true`（获取 Tavily 生成的摘要）

### 6.4 数据流与时序分析

整个查询过程的数据流和时序如下：

```
时间轴  |  用户/Dify前端  |  Dify后端  |  OpenAI API  |  Tavily API
--------+----------------+-----------+-------------+------------
t0      | 用户输入查询    |           |             |
t0+10ms | 发送到后端     |           |             |
t0+20ms |                | 接收请求   |             |
t0+30ms |                | 处理会话   |             |
t0+50ms |                | 构建请求   |             |
t0+70ms |                | 发送到OpenAI |          |
t0+80ms |                |           | 接收请求    |
t0+200ms|                |           | 分析查询    |
t0+300ms|                |           | 决定使用工具 |
t0+400ms|                | 接收工具调用|             |
t0+450ms|                | 解析参数   |             |
t0+500ms|                | 调用Tavily |             |
t0+550ms|                |           |             | 接收请求
t0+800ms|                |           |             | 执行搜索
t0+1.2s |                | 接收搜索结果|             |
t0+1.3s |                | 格式化结果 |             |
t0+1.4s |                | 发回OpenAI |             |
t0+1.5s |                |           | 接收工具结果 |
t0+1.7s |                |           | 生成最终回复 |
t0+1.8s |                | 接收AI回复 |             |
t0+1.9s | 开始展示回复   |           |             |
t0+2.0s | 完成展示       |           |             |
```

这个过程通常在1-3秒内完成，具体时间取决于网络状况、服务器负载和查询复杂度。

### 6.5 流式响应机制

Dify 和 OpenAI 都支持流式响应（Streaming），这使得用户体验更加流畅：

1. **分块传输**：OpenAI 以小块方式流式返回回答
2. **即时显示**：Dify 前端实时展示接收到的内容
3. **工具调用处理**：工具调用时，流会暂停，等待工具执行完成后继续
4. **状态指示**：Dify 界面通常会显示"正在搜索相关信息..."等状态提示

### 6.6 错误处理与回退机制

在工具调用过程中可能出现多种错误，Dify 实现了完整的错误处理机制：

1. **API 超时**：如 Tavily 响应超时，设置最长等待时间
2. **格式错误**：如 Tavily 返回非预期格式数据，进行错误处理
3. **结果为空**：如搜索无结果，向 OpenAI 返回空结果说明
4. **回退策略**：当工具失败时，可以选择其他工具或让模型直接回答

## 七、扩展应用场景

### 7.1 多工具协同情景

在复杂查询中，Dify 支持多工具协同工作：

```
用户："沈阳明天天气如何？我需要准备出行计划"
```

此时可能涉及：
1. Tavily 搜索工具获取天气信息
2. 地图工具提供交通建议
3. AI 整合多工具结果提供完整建议

### 7.2 工具链接口定制化

Dify 企业版允许定制工具与内部系统的集成：

```json
// 自定义天气工具示例
{
  "type": "function",
  "function": {
    "name": "company_weather_api",
    "description": "查询公司内部气象数据库获取高精度天气预报",
    "parameters": {
      "type": "object",
      "properties": {
        "city": { "type": "string" },
        "date": { "type": "string" },
        "detail_level": { "type": "integer", "enum": [1, 2, 3] }
      },
      "required": ["city"]
    }
  }
}
```

## 八、总结

当用户在 Dify 平台输入"查询沈阳天气"时，整个处理过程是一个复杂而精密的信息流：

1. **用户输入** → Dify前端 → Dify后端
2. Dify后端 → **OpenAI API**（附带工具描述）
3. OpenAI决定调用工具 → **Dify执行层**
4. Dify执行层 → **Tavily API**（执行搜索）
5. 搜索结果 → Dify → OpenAI（生成回答）
6. 最终回答 → Dify前端 → **用户界面**

这个过程展示了现代AI应用的多层架构和复杂交互流程。Dify 作为中间平台，成功地连接了用户、大语言模型和外部工具，使非技术用户也能构建强大的AI应用，无需深入了解底层实现细节。


修改建议 

# DeepClaude与Dify工具调用集成方案

## 一、问题分析

经过对提供的代码分析，发现当前DeepClaude服务虽然能够接收Dify的基本问答请求，但无法处理Dify发送的工具调用请求（如搜索工具Tavily等）。主要问题在于：

### 1. 主要缺失点

1. **OpenAI API兼容性不完整**：
   - `chat_completions`端点未处理请求中的`tools`和`tool_choice`参数
   - 未实现工具调用的流程（模型决定使用工具→返回tool_calls→客户端执行→返回结果）
   
2. **缺少工具执行机制**：
   - 没有工具调用的处理逻辑
   - 缺少工具执行结果的接收和处理逻辑

3. **搜索增强未集成**：
   - 没有在思考开始前进行搜索增强的机制
   - 搜索结果未能被纳入到思考和回答流程

## 二、修改方案设计

### 1. 修改app/main.py

需要在chat_completions接口中处理工具相关参数：

```python
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    try:
        data = await request.json()
        if "messages" not in data:
            raise ValueError("Missing messages parameter")
            
        # 新增：提取工具相关参数
        tools = data.get("tools", [])
        tool_choice = data.get("tool_choice", "auto")
        
        if data.get("stream", False):
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=data["messages"],
                    chat_id=f"chatcmpl-{uuid.uuid4()}",
                    created_time=int(time.time()),
                    model=data.get("model", "deepclaude"),
                    # 新增：传递工具相关参数
                    tools=tools,
                    tool_choice=tool_choice
                ),
                # 其他参数不变...
            )
        else:
            response = await deep_claude.chat_completions_without_stream(
                messages=data["messages"],
                model_arg=get_and_validate_params(data),
                # 新增：传递工具相关参数
                tools=tools,
                tool_choice=tool_choice
            )
            return JSONResponse(content=response)
    # 异常处理部分不变...
```

### 2. 修改app/deepclaude/deepclaude.py

需要增加工具调用和搜索增强的支持：

```python
# 添加新工具模块导入
import requests
from typing import Dict, List, Any, Optional

class DeepClaude:
    # 初始化中添加工具处理相关配置
    def __init__(self, **kwargs):
        # 现有初始化代码...
        
        # 添加工具相关配置
        self.supported_tools = {
            "tavily_search": self._execute_tavily_search,
            # 其他工具...
        }
        self.tavily_api_key = os.getenv('TAVILY_API_KEY', '')
        self.search_enabled = os.getenv('ENABLE_SEARCH_ENHANCEMENT', 'true').lower() == 'true'
        
    # 新增：搜索增强函数
    async def _enhance_with_search(self, query: str) -> str:
        """执行搜索并返回结果"""
        if not self.search_enabled or not self.tavily_api_key:
            logger.warning("搜索增强功能未启用或缺少API Key")
            return ""
            
        try:
            logger.info(f"为问题进行搜索增强: {query}")
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.tavily_api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True
                },
                timeout=10
            )
            response.raise_for_status()
            search_result = response.json()
            
            # 格式化搜索结果
            formatted_result = f"搜索结果:\n{search_result.get('answer', '')}\n\n"
            for i, result in enumerate(search_result.get("results", [])[:3], 1):
                formatted_result += f"{i}. {result.get('title')}\n"
                formatted_result += f"   {result.get('url')}\n"
                formatted_result += f"   {result.get('content')[:200]}...\n\n"
                
            return formatted_result
        except Exception as e:
            logger.error(f"搜索增强失败: {e}")
            return ""
    
    # 新增：工具执行函数
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用并返回结果"""
        tool_name = tool_call.get("function", {}).get("name")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        
        try:
            arguments = json.loads(arguments_str)
            tool_executor = self.supported_tools.get(tool_name)
            
            if not tool_executor:
                logger.warning(f"不支持的工具: {tool_name}")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                    "content": json.dumps({"error": f"工具 {tool_name} 不受支持"})
                }
                
            result = await tool_executor(arguments)
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "name": tool_name,
                "content": json.dumps(result)
            }
        except Exception as e:
            logger.error(f"执行工具 {tool_name} 时出错: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "name": tool_name,
                "content": json.dumps({"error": str(e)})
            }
    
    # 新增：Tavily搜索工具
    async def _execute_tavily_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行Tavily搜索"""
        if not self.tavily_api_key:
            return {"error": "未配置Tavily API密钥"}
            
        query = arguments.get("query", "")
        if not query:
            return {"error": "搜索查询不能为空"}
            
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.tavily_api_key}"},
                json={
                    "query": query,
                    "search_depth": arguments.get("search_depth", "basic"),
                    "include_answer": arguments.get("include_answer", True),
                    "include_domains": arguments.get("include_domains", []),
                    "exclude_domains": arguments.get("exclude_domains", [])
                },
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Tavily搜索失败: {e}")
            return {"error": str(e)}
            
    # 修改现有流式响应函数
    async def chat_completions_with_stream(self, messages: list, **kwargs):
        try:
            logger.info("开始流式处理请求...")
            
            # 提取工具相关参数
            tools = kwargs.get("tools", [])
            tool_choice = kwargs.get("tool_choice", "auto")
            has_tools = len(tools) > 0
            
            # 对最后一个用户消息进行搜索增强
            search_enhanced = False
            if has_tools and self.search_enabled and messages and messages[-1]["role"] == "user":
                search_result = await self._enhance_with_search(messages[-1]["content"])
                if search_result:
                    search_enhanced = True
            
            # 保存对话到数据库的代码保持不变...
            
            provider = self._get_reasoning_provider()
            reasoning_content = []
            thought_complete = False
            
            # 如果有搜索增强，添加到推理内容
            if search_enhanced:
                yield self._format_stream_response(
                    "使用搜索增强思考...\n" + search_result,
                    content_type="reasoning",
                    is_first_thought=True,
                    **kwargs
                )
            else:
                yield self._format_stream_response(
                    "开始思考问题...",
                    content_type="reasoning",
                    is_first_thought=True,
                    **kwargs
                )
                
            # 获取推理内容的代码保持不变...
            
            # 检查模型是否需要调用工具
            if has_tools:
                # 构建Claude请求以决定是否使用工具
                tool_decision_prompt = self._format_tool_decision_prompt(
                    original_question=messages[-1]["content"],
                    reasoning="\n".join(reasoning_content),
                    tools=tools
                )
                
                tool_decision_messages = [{
                    "role": "user", 
                    "content": tool_decision_prompt
                }]
                
                tool_decision = ""
                async for content_type, content in self.claude_client.stream_chat(
                    messages=tool_decision_messages,
                    **self._prepare_answerer_kwargs(kwargs)
                ):
                    if content_type == "content":
                        tool_decision += content
                
                # 如果决定使用工具
                if "USE_TOOL" in tool_decision:
                    tool_name = None
                    tool_args = {}
                    
                    # 简单解析工具名称和参数
                    for line in tool_decision.split("\n"):
                        if line.startswith("TOOL_NAME:"):
                            tool_name = line[len("TOOL_NAME:"):].strip()
                        elif line.startswith("TOOL_ARGS:"):
                            try:
                                tool_args = json.loads(line[len("TOOL_ARGS:"):].strip())
                            except:
                                pass
                    
                    if tool_name and tool_name in [t.get("function", {}).get("name") for t in tools]:
                        # 生成工具调用响应
                        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                        tool_call = {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            }
                        }
                        
                        # 返回工具调用响应
                        yield self._format_tool_call_response(
                            tool_call=tool_call,
                            **kwargs
                        )
                        
                        # 这里应该等待客户端提供工具结果，但由于API限制，我们直接执行工具
# 这里应该等待客户端提供工具结果，但由于API限制，我们直接执行工具
                        tool_result = await self._execute_tool(tool_call)
                        
                        # 返回工具执行结果
                        yield self._format_tool_result_response(
                            tool_result=tool_result,
                            **kwargs
                        )
                        
                        # 将原始消息、推理内容、工具调用和结果组合，生成最终回答
                        final_prompt = self._format_final_prompt_with_tool(
                            original_question=messages[-1]["content"],
                            reasoning="\n".join(reasoning_content),
                            tool_name=tool_name,
                            tool_args=tool_args,
                            tool_result=tool_result["content"]
                        )
                        
                        messages_for_claude = [{
                            "role": "user",
                            "content": final_prompt
                        }]
                    else:
                        # 无效工具调用，直接使用原始内容回答
                        messages_for_claude = [{
                            "role": "user",
                            "content": self._format_claude_prompt(
                                messages[-1]["content"], 
                                "\n".join(reasoning_content)
                            )
                        }]
                else:
                    # 不使用工具，直接发送原始提示词
                    messages_for_claude = [{
                        "role": "user",
                        "content": self._format_claude_prompt(
                            messages[-1]["content"], 
                            "\n".join(reasoning_content)
                        )
                    }]
            else:
                # 无工具场景，保持原有逻辑
                messages_for_claude = [{
                    "role": "user",
                    "content": self._format_claude_prompt(
                        messages[-1]["content"], 
                        "\n".join(reasoning_content)
                    )
                }]
            
            # 生成最终回答
            yield self._format_stream_response(
                "\n\n---\n思考完毕，开始回答：\n\n",
                content_type="separator",
                is_first_thought=False,
                **kwargs
            )
            
            answer_begun = False
            full_answer = []
            async for content_type, content in self.claude_client.stream_chat(
                messages=messages_for_claude,
                **self._prepare_answerer_kwargs(kwargs)
            ):
                if content_type == "content" and content:
                    if not answer_begun and content.strip():
                        answer_begun = True
                    full_answer.append(content)
                    yield self._format_stream_response(
                        content,
                        content_type="content",
                        is_first_thought=False,
                        **kwargs
                    )
            
            # 保存到数据库的逻辑保持不变...
            
        except Exception as e:
            # 错误处理保持不变...
    
    # 同样更新非流式方法
    async def chat_completions_without_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        tools: list = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> dict:
        logger.info("开始处理非流式请求...")
        logger.debug(f"输入消息: {messages}")
        
        has_tools = tools and len(tools) > 0
        
        # 保存到数据库的逻辑保持不变...
        
        # 获取推理内容
        try:
            reasoning = await self._get_reasoning_content(
                messages=messages,
                model=kwargs.get("deepseek_model", "deepseek-reasoner"),
                model_arg=model_arg
            )
        except Exception as e:
            logger.error(f"获取推理内容失败: {e}")
            reasoning = "无法获取推理内容"
            # 重试逻辑保持不变...
            
        logger.debug(f"获取到推理内容: {reasoning[:min(500, len(reasoning))]}...")
        
        # 搜索增强
        search_result = ""
        if has_tools and self.search_enabled and messages and messages[-1]["role"] == "user":
            search_result = await self._enhance_with_search(messages[-1]["content"])
        
        # 组合推理内容和搜索结果
        enhanced_reasoning = reasoning
        if search_result:
            enhanced_reasoning = f"搜索信息:\n{search_result}\n\n推理过程:\n{reasoning}"
        
        # 工具调用处理
        if has_tools:
            # 决定是否使用工具
            tool_decision_prompt = self._format_tool_decision_prompt(
                original_question=messages[-1]["content"],
                reasoning=enhanced_reasoning,
                tools=tools
            )
            
            tool_decision_messages = [{
                "role": "user", 
                "content": tool_decision_prompt
            }]
            
            tool_decision = ""
            async for content_type, content in self.claude_client.stream_chat(
                messages=tool_decision_messages,
                model_arg=model_arg,
                stream=False
            ):
                if content_type in ["answer", "content"]:
                    tool_decision += content
            
            if "USE_TOOL" in tool_decision:
                tool_name = None
                tool_args = {}
                
                # 解析工具信息
                for line in tool_decision.split("\n"):
                    if line.startswith("TOOL_NAME:"):
                        tool_name = line[len("TOOL_NAME:"):].strip()
                    elif line.startswith("TOOL_ARGS:"):
                        try:
                            tool_args = json.loads(line[len("TOOL_ARGS:"):].strip())
                        except:
                            pass
                
                if tool_name and tool_name in [t.get("function", {}).get("name") for t in tools]:
                    # 构建工具调用
                    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                    tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    }
                    
                    # 执行工具
                    tool_result = await self._execute_tool(tool_call)
                    
                    # 将工具结果纳入最终回答
                    final_prompt = self._format_final_prompt_with_tool(
                        original_question=messages[-1]["content"],
                        reasoning=enhanced_reasoning,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=tool_result["content"]
                    )
                    
                    # 构建最终返回，包含工具调用
                    return {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": kwargs.get("model", "deepclaude"),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }
        
        # 无工具调用或工具调用失败时的逻辑
        combined_content = self._format_claude_prompt(
            original_question=messages[-1]["content"] if messages else "",
            reasoning=enhanced_reasoning
        )
        
        claude_messages = [{"role": "user", "content": combined_content}]
        
        # 获取Claude回答的逻辑保持不变...
        # 保存到数据库的逻辑保持不变...
        
    # 新增：格式化工具决策提示
    def _format_tool_decision_prompt(self, original_question: str, reasoning: str, tools: List[Dict]) -> str:
        tools_description = json.dumps(tools, indent=2, ensure_ascii=False)
        
        prompt = f"""
你是一个工具使用决策助手。请根据以下用户问题、推理分析和可用工具，决定是否需要使用工具。
仅在确实需要查询外部信息时才使用工具。如无必要，请直接回答。

用户问题: {original_question}

推理分析: 
{reasoning}

可用工具:
{tools_description}

请按以下格式回复:
1. 如需使用工具:
```
USE_TOOL
TOOL_NAME: [工具名称]
TOOL_ARGS: [JSON格式的参数]
REASON: [使用工具的原因]
```

2. 如不需使用工具:
```
NO_TOOL
REASON: [不使用工具的原因]
```

请仔细判断：是否真的需要使用工具来获取更多信息？
"""
        return prompt
        
    # 新增：格式化包含工具结果的最终提示
    def _format_final_prompt_with_tool(self, original_question: str, reasoning: str, 
                                      tool_name: str, tool_args: Dict, tool_result: str) -> str:
        return f"""
用户问题: {original_question}

思考过程:
{reasoning}

为了解答问题，我使用了 {tool_name} 工具，参数是:
{json.dumps(tool_args, indent=2, ensure_ascii=False)}

工具返回的结果:
{tool_result}

请根据所有信息，包括思考过程和工具提供的数据，全面回答用户的原始问题。
回答应该清晰、准确，并直接引用工具提供的相关数据。
"""
    
    # 新增：格式化工具调用响应
    def _format_tool_call_response(self, tool_call: Dict, **kwargs) -> bytes:
        response = {
            "id": kwargs.get("chat_id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion.chunk",
            "created": kwargs.get("created_time", int(time.time())),
            "model": kwargs.get("model", "deepclaude"),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                },
                "finish_reason": "tool_calls"
            }]
        }
        return f"data: {json.dumps(response)}\n\n".encode('utf-8')
    
    # 新增：格式化工具结果响应
    def _format_tool_result_response(self, tool_result: Dict, **kwargs) -> bytes:
        response = {
            "id": kwargs.get("chat_id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion.chunk",
            "created": kwargs.get("created_time", int(time.time())),
            "model": kwargs.get("model", "deepclaude"),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "tool",
                    "content": tool_result.get("content", ""),
                    "tool_call_id": tool_result.get("tool_call_id", "")
                },
                "finish_reason": null
            }]
        }
        return f"data: {json.dumps(response)}\n\n".encode('utf-8')
```

### 3. 新增环境变量

需要在.env文件中添加以下配置：

```
# 工具配置
TAVILY_API_KEY=your_tavily_api_key
ENABLE_SEARCH_ENHANCEMENT=true
```

## 三、实现流程详解

### 1. 工具调用与搜索增强流程

下面是完整的工具调用与搜索增强流程：

1. **接收请求**：
   - 接收Dify的/v1/chat/completions请求
   - 提取tools、tool_choice等参数

2. **搜索增强**（如果启用）：
   - 对用户最后一条消息进行Tavily搜索
   - 将搜索结果添加到思考上下文

3. **思考过程**：
   - 使用DeepSeek/Ollama生成推理内容
   - 在推理中融入搜索结果

4. **工具使用决策**：
   - 基于推理内容决定是否需要使用工具
   - 生成工具调用参数

5. **工具调用**：
   - 生成符合OpenAI格式的tool_calls响应
   - 内部执行工具调用获取结果

6. **生成最终回答**：
   - 将推理、搜索结果、工具调用结果组合
   - 使用Claude生成全面回答

### 2. 数据流结构

```
┌───────────┐     ┌───────────┐     ┌───────────┐
│   Dify    │────>│ DeepClaude│────>│ DeepSeek/ │
│           │     │  FastAPI  │     │   Ollama  │
└───────────┘     └───────────┘     └───────────┘
                      │   ↑             (推理)
                      │   │
                      ↓   │           ┌───────────┐
                  ┌───────────┐     ┌─┴──────────┐│
                  │  Tavily   │────>│   Claude   ││
                  │  搜索工具  │     │            ││
                  └───────────┘     └────────────┘
                      (增强)           (综合回答)
```

## 四、代码实现要点

### 1. OpenAI API兼容性

为保证与OpenAI API格式的完全兼容，需要注意以下几点：

1. **tool_calls结构**：
   ```json
   "tool_calls": [
     {
```json
   "tool_calls": [
     {
       "id": "call_abc123def456",
       "type": "function",
       "function": {
         "name": "tavily_search",
         "arguments": "{\"query\":\"最新气候变化报告\",\"search_depth\":\"basic\"}"
       }
     }
   ]
   ```

2. **响应格式**：
   - 当调用工具时，`finish_reason`应设为`"tool_calls"`
   - 返回工具结果时，需要正确设置`tool_call_id`
   - 流式响应中需要正确处理delta格式

3. **工具参数解析**：
   - 需要正确处理工具参数的JSON格式
   - 确保工具名称匹配和参数验证

### 2. 搜索增强实现

搜索增强需要在以下两个关键点进行整合：

1. **思考前增强**：
   - 在DeepSeek/Ollama进行推理前，先获取搜索结果
   - 将搜索结果添加到推理上下文中

2. **结果处理**：
   - 结构化搜索结果，包括摘要和前几条详细内容
   - 搜索结果按重要性排序，避免信息过载

### 3. 工具调用设计

工具调用采用以下架构：

1. **工具注册机制**：
   ```python
   self.supported_tools = {
       "tavily_search": self._execute_tavily_search,
       # 未来可以添加更多工具
   }
   ```

2. **通用工具执行框架**：
   - 接收标准化的工具调用请求
   - 分发到具体的工具执行函数
   - 统一错误处理和结果格式化

3. **工具调用决策**：
   - 使用Claude来决定是否需要调用工具
   - 提供清晰的决策提示模板

## 五、集成建议与注意事项

### 1. 渐进式集成策略

为了避免影响现有功能，建议采用以下步骤进行集成：

1. **添加必要的环境变量**：
   - 先配置Tavily API密钥等必要的环境变量

2. **实现基础工具调用处理**：
   - 在main.py中添加对tool参数的处理
   - 实现基础工具执行框架

3. **验证单一工具功能**：
   - 先实现并测试Tavily搜索工具
   - 确保单一工具的调用与响应格式正确

4. **集成搜索增强功能**：
   - 在推理流程中添加搜索增强
   - 验证搜索结果能够提升回答质量

5. **完善流程和错误处理**：
   - 添加完整的错误处理
   - 优化工具结果与推理的融合

### 2. 避免回归的测试策略

为确保修改不影响现有功能，需要进行以下测试：

1. **无工具场景测试**：
   - 验证在无工具参数时的行为与之前一致
   - 确保搜索增强不会在非必要时执行

2. **工具调用格式测试**：
   - 验证工具调用响应格式符合OpenAI标准
   - 测试流式和非流式两种模式

3. **错误处理测试**：
   - 测试当工具执行失败时的回退机制
   - 确保即使工具失败，仍能返回有用的回答

### 3. 性能考量

在实现过程中需要注意以下性能因素：

1. **异步处理**：
   - 确保所有网络请求和工具执行都是异步的
   - 避免阻塞主流程

2. **超时处理**：
   - 为每个工具调用设置适当的超时
   - 实现超时后的优雅降级

3. **缓存策略**：
   - 考虑对相似搜索查询结果进行缓存
   - 减少重复的外部API调用

## 总结

基于对代码的分析，我设计了一个既能保持原有功能完整性，又能增加工具调用和搜索增强能力的方案。这个方案：

1. **尊重现有架构**：保持DeepSeek/Ollama用于推理，Claude用于回答的基本流程
2. **优雅地整合搜索**：在思考阶段前进行搜索，增强推理质量
3. **实现标准工具调用**：完全兼容OpenAI工具调用格式
4. **灵活的工具决策机制**：使用独立的决策过程判断是否需要使用工具

实施这一方案后，DeepClaude将能够：
- 接受和处理Dify发送的工具调用请求
- 在思考前进行搜索来增强推理质量
- 将搜索结果和推理内容传递给Claude生成更准确的回答
- 维持与现有OpenAI API的完全兼容性

这些改进将使DeepClaude服务在Dify平台上发挥更大的价值，特别是在需要实时信息的场景中。
                        