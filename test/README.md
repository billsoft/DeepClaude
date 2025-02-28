# DeepClaude 测试脚本使用说明

本目录包含多个测试脚本，用于测试不同API提供商的大语言模型能力。

## 配置环境变量

在运行测试脚本前，请确保已正确配置环境变量。您可以通过以下两种方式之一配置环境变量：

1. 在项目根目录的`.env`文件中设置
2. 通过命令行参数传递（仅限部分测试脚本支持）

### .env文件配置示例

```env
# DeepSeek 官方配置
DEEPSEEK_API_KEY=sk-xxxxxx
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner
DEEPSEEK_PROVIDER=deepseek

# 硅基云配置
# DEEPSEEK_API_KEY=sk-xxxxxx
# DEEPSEEK_API_URL=https://api.siliconflow.cn/v1/chat/completions
# DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1
# DEEPSEEK_PROVIDER=siliconflow

# NVIDIA 配置
# DEEPSEEK_API_KEY=nvapi-xxxxxx
# DEEPSEEK_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
# DEEPSEEK_MODEL=deepseek-ai/deepseek-r1
# DEEPSEEK_PROVIDER=nvidia

# Claude 配置
CLAUDE_API_KEY=sk-ant-api03-xxxxxx
CLAUDE_API_URL=https://claude.agentpt.com/v1/messages
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_PROVIDER=anthropic

# 推理模式配置
REASONING_PROVIDER=deepseek
IS_ORIGIN_REASONING=true
DEEPSEEK_REASONING_MODE=reasoning_field
```

## 可用测试脚本

### 1. 测试 DeepSeek API

```bash
python test/test_deepseek_client.py --provider deepseek --reasoning-mode auto --question "计算1+1等于几?"
```

### 2. 测试 NVIDIA API

```bash
python test/test_nvidia_deepseek.py
```

### 3. 测试 硅基流动 API (SiliconFlow)

使用命令行参数直接提供API密钥（推荐）：

```bash
python test/test_siliconflow_deepseek.py --api-key YOUR_API_KEY --question "中国大模型行业未来展望如何？"
```

或者使用环境变量（需要修改.env文件）：

```bash
# 先在.env文件中取消硅基云配置部分的注释并设置正确的API密钥
python test/test_siliconflow_deepseek.py
```

该测试脚本会进行两项测试：
1. 流式API调用测试，验证推理过程和回答内容的实时生成
2. 非流式API调用测试，验证完整的响应结构

注意：运行硅基流动测试前，请取消`.env`文件中硅基云配置部分的注释，或确保已正确设置`DEEPSEEK_API_KEY`环境变量。

### 4. 测试 Claude API

```bash
python test/test_claude_client.py
```

### 5. 测试 Ollama R1

```bash
python test/test_ollama_r1.py
```

注意：运行Ollama测试前，请确保Ollama服务正在运行，并已下载`deepseek-r1`模型。

## 测试输出

所有测试脚本都会输出详细的日志信息，包括：

- API配置信息
- 请求/响应数据
- 推理内容和回答内容
- 错误信息（如有）

## 故障排除

1. 如果遇到API密钥错误，请检查`.env`文件中的配置
2. 如果未收到推理内容，请尝试调整`DEEPSEEK_REASONING_MODE`和`IS_ORIGIN_REASONING`设置
3. 如果遇到网络错误，请检查是否需要配置代理 