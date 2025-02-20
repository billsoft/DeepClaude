"""NVIDIA DeepSeek API 客户端测试"""
import os
import sys
import asyncio
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger

load_dotenv()

async def test_nvidia_deepseek_stream():
    """测试 NVIDIA DeepSeek 流式输出"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL")
    
    logger.info("=== NVIDIA DeepSeek 客户端测试开始 ===")
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key 是否存在: {bool(api_key)}")
    
    if not api_key:
        logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return
        
    messages = [
        {"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}
    ]
    
    client = DeepSeekClient(
        api_key=api_key,
        api_url=api_url,
        provider="nvidia"
    )
    
    try:
        logger.info("开始测试 NVIDIA DeepSeek 流式输出...")
        logger.debug(f"发送消息: {messages}")
        
        reasoning_buffer = []
        content_buffer = []
        
        async for content_type, content in client.stream_chat(
            messages=messages,
            model="deepseek-ai/deepseek-r1"
        ):
            if content_type == "reasoning":
                reasoning_buffer.append(content)
                if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
                    reasoning_buffer = []
            elif content_type == "content":
                content_buffer.append(content)
                if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"最终答案：{''.join(content_buffer)}")
                    content_buffer = []
                    
        # 输出剩余内容
        if reasoning_buffer:
            logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
        if content_buffer:
            logger.info(f"最终答案：{''.join(content_buffer)}")
            
        logger.info("=== NVIDIA DeepSeek 客户端测试完成 ===")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)

def main():
    """主函数"""
    asyncio.run(test_nvidia_deepseek_stream())

if __name__ == "__main__":
    main() 