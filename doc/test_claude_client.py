"""Claude API 客户端测试"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

from app.clients.claude_client import ClaudeClient
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

async def test_claude_stream():
    """测试 Claude 流式输出"""
    # 从环境变量获取配置
    api_key = os.getenv("CLAUDE_API_KEY")
    api_url = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")  # 设置默认值
    provider = os.getenv("CLAUDE_PROVIDER", "anthropic")
    
    # 添加配置信息日志
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key 是否存在: {bool(api_key)}")
    logger.info(f"Provider: {provider}")
    
    if not api_key:
        logger.error("请在 .env 文件中设置 CLAUDE_API_KEY")
        return
        
    # 创建测试消息
    messages = [
        {"role": "user", "content": "9.8和9.111谁大"}
    ]
    
    # 初始化客户端
    client = ClaudeClient(api_key, api_url, provider)
    
    try:
        logger.info("开始测试 Claude 流式输出...")
        # 使用与 deepclaude.py 相同的参数设置
        async for content_type, content in client.stream_chat(
            messages=messages,
            model_arg=(0.7, 0.9, 0, 0),  # temperature, top_p, presence_penalty, frequency_penalty
            model="claude-3-5-sonnet-20241022"
        ):
            if content_type == "answer":
                logger.info(f"收到回答内容: {content}")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        logger.error(f"错误类型: {type(e)}")  # 添加错误类型信息

def main():
    """主函数"""
    asyncio.run(test_claude_stream())

if __name__ == "__main__":
    main() 