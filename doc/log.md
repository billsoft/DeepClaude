# 下面是从启动到接收请求再到返回响应的日志
(DeepClaude) (base) wanglei@wangleideMac-mini DeepClaude % uvicorn app.main:app

2025-02-15 14:04:37 - DeepClaude - INFO - 当前工作目录: /Users/wanglei/code/DeepClaude
2025-02-15 14:04:37 - DeepClaude - INFO - 尝试加载.env文件...
2025-02-15 14:04:37 - DeepClaude - INFO - ALLOW_API_KEY环境变量状态: 已设置
2025-02-15 14:04:37 - DeepClaude - INFO - Loaded API key starting with: Deep
2025-02-15 14:04:37 - DeepClaude - INFO - 开始请求
INFO:     Started server process [4511]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:63862 - "OPTIONS /v1/chat/completions HTTP/1.1" 200 OK
2025-02-15 16:40:21 - DeepClaude - INFO - API密钥验证通过
INFO:     127.0.0.1:63862 - "POST /v1/chat/completions HTTP/1.1" 200 OK
2025-02-15 16:40:21 - DeepClaude - INFO - 开始处理 DeepSeek 流，使用模型：deepseek-reasoner, 提供商: deepseek
2025-02-15 16:40:21 - DeepClaude - INFO - 等待获取 DeepSeek 的推理内容...
2025-02-15 16:40:59 - DeepClaude - INFO - 提取内容信息，推理阶段结束: 以下是
2025-02-15 16:40:59 - DeepClaude - INFO - DeepSeek 推理完成，收集到的推理内容长度：1231
2025-02-15 16:40:59 - DeepClaude - INFO - DeepSeek 任务处理完成，标记结束
2025-02-15 16:40:59 - DeepClaude - INFO - 开始处理 Claude 流，使用模型: claude-3-5-sonnet-20241022, 提供商: anthropic
2025-02-15 16:41:12 - DeepClaude - INFO - Claude 任务处理完成，标记结束
2025-02-15 16:41:12 - DeepClaude - INFO - API密钥验证通过
