import os
import sys
import asyncio
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.clients.ollama_r1 import OllamaR1Client
from app.utils.logger import logger

load_dotenv()
os.environ['LOG_LEVEL'] = 'DEBUG'

async def test_ollama_stream():
    api_url = os.getenv("OLLAMA_API_URL", "http://192.168.100.81:11434/api/chat")
    logger.info(f"API URL: {api_url}")
    
    client = OllamaR1Client(api_url)
    try:
        messages = [
            {"role": "user", "content": "9.9和9.11谁大?"}
        ]
        logger.info("开始测试 Ollama R1 流式输出...")
        logger.debug(f"发送消息: {messages}")
        
        async for msg_type, content in client.stream_chat(messages):
            if msg_type == "reasoning":
                logger.info(f"推理过程: {content}")
            else:
                logger.info(f"最终答案: {content}")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_ollama_stream())