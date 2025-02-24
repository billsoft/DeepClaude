"""DeepSeek API 客户端测试"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

# 设置日志级别为 DEBUG
os.environ['LOG_LEVEL'] = 'DEBUG'

async def test_deepseek_stream():
    """测试 DeepSeek 流式输出"""
    # 从环境变量获取配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    is_origin_reasoning = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"
    
    # 只在开始时输出配置信息
    logger.info("=== DeepSeek 客户端测试开始 ===")
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key 是否存在: {bool(api_key)}")
    logger.info(f"原始推理模式: {is_origin_reasoning}")
    
    if not api_key:
        logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return
        
    messages = [
        {"role": "user", "content": "1+1等于几?"}
    ]
    
    # 初始化客户端
    client = DeepSeekClient(api_key, api_url)
    
    try:
        logger.info("开始测试 DeepSeek 流式输出...")
        logger.debug(f"发送消息: {messages}")
        
        reasoning_buffer = []
        content_buffer = []
        
        async for content_type, content in client.stream_chat(
            messages=messages,
            model="deepseek-ai/DeepSeek-R1",
            is_origin_reasoning=is_origin_reasoning
        ):
            if content_type == "reasoning":
                reasoning_buffer.append(content)
                # 当收集到一定数量的字符或遇到标点符号时输出
                if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
                    reasoning_buffer = []
            elif content_type == "content":
                content_buffer.append(content)
                if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"最终答案：{''.join(content_buffer)}")
                    content_buffer = []
                
        # 输出剩余的内容
        if reasoning_buffer:
            logger.debug(f"推理过程：{''.join(reasoning_buffer)}")
        if content_buffer:
            logger.info(f"最终答案：{''.join(content_buffer)}")
            
        logger.info("=== DeepSeek 客户端测试完成 ===")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        logger.error(f"错误类型: {type(e)}")

def main():
    """主函数"""
    asyncio.run(test_deepseek_stream())

if __name__ == "__main__":
    main() 