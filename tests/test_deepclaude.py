import os
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock
from app.deepclaude.core import DeepClaude
from app.deepclaude.reasoning.factory import ReasoningProviderFactory
from app.deepclaude.tools.handlers import ToolHandler

@pytest.fixture
def test_env_setup():
    """设置测试环境变量"""
    os.environ["CLAUDE_API_KEY"] = "test_claude_key"
    os.environ["CLAUDE_MODEL"] = "claude-3-7-sonnet-20250219"
    os.environ["REASONING_PROVIDER"] = "deepseek"
    os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
    os.environ["SAVE_TO_DB"] = "false"
    
    # 使用完成后恢复环境变量
    original_values = {
        "CLAUDE_API_KEY": os.environ.get("CLAUDE_API_KEY"),
        "CLAUDE_MODEL": os.environ.get("CLAUDE_MODEL"),
        "REASONING_PROVIDER": os.environ.get("REASONING_PROVIDER"),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
        "SAVE_TO_DB": os.environ.get("SAVE_TO_DB")
    }
    
    yield
    
    # 恢复原始环境变量
    for key, value in original_values.items():
        if value is not None:
            os.environ[key] = value
        else:
            if key in os.environ:
                del os.environ[key]

@pytest.mark.asyncio
async def test_deepclaude_initialization(test_env_setup):
    """测试DeepClaude初始化"""
    # 初始化DeepClaude实例
    deepclaude = DeepClaude()
    
    # 验证基本属性
    assert deepclaude.claude_api_key == "test_claude_key"
    assert deepclaude.claude_provider == "anthropic"
    assert deepclaude.save_to_db is False
    assert deepclaude.min_reasoning_chars == 100
    
    # 验证子组件
    assert deepclaude.claude_client is not None
    assert deepclaude.tool_handler is not None
    assert deepclaude.thinker_client is not None

@pytest.mark.asyncio
async def test_format_tool_decision_prompt():
    """测试格式化工具决策提示"""
    deepclaude = DeepClaude()
    
    # 测试函数类型工具
    tools = [
        {
            "function": {
                "name": "test_function",
                "description": "测试函数描述",
                "parameters": {
                    "type": "object",
                    "required": ["required_param"],
                    "properties": {
                        "required_param": {
                            "type": "string",
                            "description": "必填参数描述"
                        },
                        "optional_param": {
                            "type": "integer",
                            "description": "可选参数描述",
                            "enum": [1, 2, 3]
                        }
                    }
                }
            }
        }
    ]
    
    prompt = deepclaude._format_tool_decision_prompt(
        original_question="测试问题",
        reasoning="测试推理过程",
        tools=tools
    )
    
    # 验证提示内容
    assert "测试问题" in prompt
    assert "测试推理过程" in prompt
    assert "test_function" in prompt
    assert "测试函数描述" in prompt
    assert "required_param" in prompt
    assert "必填" in prompt
    assert "optional_param" in prompt
    assert "可选" in prompt
    assert "可选值: 1, 2, 3" in prompt
    
    # 测试自定义工具
    tools = [
        {
            "type": "custom",
            "name": "test_custom",
            "description": "自定义工具描述",
            "tool_schema": {
                "type": "object",
                "required": ["required_param"],
                "properties": {
                    "required_param": {
                        "type": "string",
                        "description": "必填参数描述"
                    }
                }
            }
        }
    ]
    
    prompt = deepclaude._format_tool_decision_prompt(
        original_question="测试问题",
        reasoning="测试推理过程",
        tools=tools
    )
    
    # 验证自定义工具提示
    assert "test_custom" in prompt
    assert "自定义工具描述" in prompt
    assert "required_param" in prompt
    assert "必填" in prompt

@pytest.mark.asyncio
@patch("app.deepclaude.core.ReasoningProviderFactory.create")
@patch("app.clients.claude_client.ClaudeClient.chat")
async def test_chat_completions_without_stream(mock_claude_chat, mock_reasoning_factory, test_env_setup):
    """测试非流式聊天完成功能"""
    # 设置模拟对象的返回值
    mock_thinker = MagicMock()
    mock_thinker.get_reasoning.return_value = "模拟推理结果"
    mock_reasoning_factory.return_value = mock_thinker
    
    mock_claude_chat.return_value = {"content": "模拟Claude回答"}
    
    # 创建DeepClaude实例
    deepclaude = DeepClaude()
    
    # 调用非流式聊天功能
    response = await deepclaude.chat_completions_without_stream(
        messages=[{"role": "user", "content": "测试问题"}],
        model_arg=(0.7, 0.9)
    )
    
    # 验证响应
    assert response["content"] == "模拟Claude回答"
    assert response["role"] == "assistant"
    assert response["reasoning"] == "模拟推理结果"
    
    # 验证正确调用了推理提供者
    mock_thinker.get_reasoning.assert_called_once()
    
    # 验证正确调用了Claude客户端
    mock_claude_chat.assert_called_once()
    call_args = mock_claude_chat.call_args[1]
    assert "我已经思考了以下问题" in call_args["messages"][0]["content"]
    assert "模拟推理结果" in call_args["messages"][0]["content"]

@pytest.mark.asyncio
@patch("app.deepclaude.core.ReasoningProviderFactory.create")
@patch("app.deepclaude.tools.handlers.ToolHandler.validate_and_convert_tools")
@patch("app.clients.claude_client.ClaudeClient.chat")
async def test_direct_tool_pass_without_stream(mock_claude_chat, mock_validate_tools, mock_reasoning_factory, test_env_setup):
    """测试直接工具透传模式(非流式)"""
    # 设置环境变量启用直接透传模式
    os.environ["CLAUDE_DIRECT_TOOL_PASS"] = "true"
    
    # 设置模拟对象的返回值
    mock_validate_tools.return_value = [{"function": {"name": "test_tool"}}]
    mock_claude_chat.return_value = {
        "content": None,
        "tool_calls": [
            {"type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
        ]
    }
    
    # 创建DeepClaude实例
    deepclaude = DeepClaude()
    
    # 调用透传模式
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    response = await deepclaude.chat_completions_without_stream(
        messages=[{"role": "user", "content": "使用工具"}],
        model_arg=(0.7, 0.9),
        tools=tools
    )
    
    # 验证响应
    assert "tool_calls" in response
    assert response["tool_calls"][0]["function"]["name"] == "test_tool"
    
    # 验证工具验证被调用
    mock_validate_tools.assert_called_once_with(tools, target_format='claude-3')
    
    # 验证Claude客户端被正确调用
    mock_claude_chat.assert_called_once()
    
    # 恢复环境变量
    del os.environ["CLAUDE_DIRECT_TOOL_PASS"] 