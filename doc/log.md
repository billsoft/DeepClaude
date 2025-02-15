# DeepClaude 系统日志说明

## 1. 服务启动阶段

```shell
(DeepClaude) (base) wanglei@wangleideMac-mini DeepClaude % uvicorn app.main:app
```

### 1.1 环境初始化
```log
2025-02-15 14:04:37 - DeepClaude - INFO - 当前工作目录: /Users/wanglei/code/DeepClaude
2025-02-15 14:04:37 - DeepClaude - INFO - 尝试加载.env文件...
2025-02-15 14:04:37 - DeepClaude - INFO - ALLOW_API_KEY环境变量状态: 已设置
2025-02-15 14:04:37 - DeepClaude - INFO - Loaded API key starting with: Deep
```

对应代码：`app/utils/auth.py` 中的环境变量加载和API密钥验证逻辑

### 1.2 服务器启动
```log
INFO:     Started server process [4511]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

对应代码：`app/main.py` 中的FastAPI应用初始化和启动

## 2. 请求处理流程

### 2.1 请求接收与验证
```log
INFO:     127.0.0.1:63862 - "OPTIONS /v1/chat/completions HTTP/1.1" 200 OK
2025-02-15 16:40:21 - DeepClaude - INFO - API密钥验证通过
INFO:     127.0.0.1:63862 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

对应代码：
- `app/main.py` 中的路由处理
- `app/utils/auth.py` 中的API密钥验证

### 2.2 DeepSeek处理流程
```log
2025-02-15 16:40:21 - DeepClaude - INFO - 开始处理 DeepSeek 流，使用模型：deepseek-reasoner, 提供商: deepseek
2025-02-15 16:40:21 - DeepClaude - INFO - 等待获取 DeepSeek 的推理内容...
2025-02-15 16:40:59 - DeepClaude - INFO - 提取内容信息，推理阶段结束: 以下是
2025-02-15 16:40:59 - DeepClaude - INFO - DeepSeek 推理完成，收集到的推理内容长度：1231
2025-02-15 16:40:59 - DeepClaude - INFO - DeepSeek 任务处理完成，标记结束
```

对应代码：
- `app/clients/deepseek_client.py` 中的DeepSeek API调用
- `app/deepclaude/deepclaude.py` 中的流式处理逻辑

### 2.3 Claude处理流程
```log
2025-02-15 16:40:59 - DeepClaude - INFO - 开始处理 Claude 流，使用模型: claude-3-5-sonnet-20241022, 提供商: anthropic
2025-02-15 16:41:12 - DeepClaude - INFO - Claude 任务处理完成，标记结束
2025-02-15 16:41:12 - DeepClaude - INFO - API密钥验证通过
```

对应代码：
- `app/clients/claude_client.py` 中的Claude API调用
- `app/deepclaude/deepclaude.py` 中的流式处理逻辑

## 3. 日志说明

### 3.1 日志格式
系统日志采用统一的格式：
```
时间戳 - 系统名称 - 日志级别 - 日志内容
```

对应代码：`app/utils/logger.py` 中的日志配置

### 3.2 关键日志节点
1. 环境初始化：记录工作目录、环境变量加载状态
2. API验证：记录API密钥验证结果
3. 请求处理：记录HTTP请求方法、路径和状态码
4. 模型调用：记录使用的模型信息和处理状态
5. 任务完成：记录处理结果和任务结束标记
