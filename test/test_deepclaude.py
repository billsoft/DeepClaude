import os

async def test_reasoning_fallback():
    """测试推理提供者失败时的回退机制"""
    deepclaude = DeepClaude(...)
    messages = [{"role": "user", "content": "测试问题"}]
    
    # 测试 DeepSeek 失败时切换到 Ollama
    os.environ['REASONING_PROVIDER'] = 'deepseek'
    reasoning = await deepclaude._get_reasoning_with_fallback(
        messages=messages,
        model="deepseek-reasoner"
    )
    assert reasoning  # 确保获取到推理内容 