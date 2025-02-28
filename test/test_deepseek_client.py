"""DeepSeek API 客户端测试"""
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试 DeepSeek 客户端')
    parser.add_argument('--reasoning-mode', type=str, choices=['auto', 'reasoning_field', 'think_tags', 'any_content'],
                        default=os.getenv('DEEPSEEK_REASONING_MODE', 'auto'),
                        help='推理内容提取模式')
    parser.add_argument('--provider', type=str, choices=['deepseek', 'siliconflow', 'nvidia'],
                        default=os.getenv('DEEPSEEK_PROVIDER', 'deepseek'),
                        help='API提供商')
    parser.add_argument('--model', type=str,
                        default=os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner'),
                        help='模型名称')
    parser.add_argument('--question', type=str, default='1+1等于几?',
                        help='测试问题')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()

async def test_deepseek_stream(args):
    """测试 DeepSeek 流式输出"""
    # 从环境变量获取配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    
    # 设置推理模式
    os.environ["DEEPSEEK_REASONING_MODE"] = args.reasoning_mode
    os.environ["DEEPSEEK_PROVIDER"] = args.provider
    
    # 只在开始时输出配置信息
    logger.info("=== DeepSeek 客户端测试开始 ===")
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key 是否存在: {bool(api_key)}")
    logger.info(f"提供商: {args.provider}")
    logger.info(f"推理模式: {args.reasoning_mode}")
    logger.info(f"使用模型: {args.model}")
    logger.info(f"测试问题: {args.question}")
    
    if not api_key:
        logger.error("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        return
        
    messages = [
        {"role": "user", "content": args.question}
    ]
    
    # 初始化客户端
    client = DeepSeekClient(api_key, api_url, provider=args.provider)
    
    try:
        logger.info("开始测试 DeepSeek 流式输出...")
        logger.debug(f"发送消息: {messages}")
        
        reasoning_buffer = []
        content_buffer = []
        
        # 统计计数器
        reasoning_count = 0
        content_count = 0
        
        async for content_type, content in client.get_reasoning(
            messages=messages,
            model=args.model
        ):
            if content_type == "reasoning":
                reasoning_count += 1
                reasoning_buffer.append(content)
                # 收集到一定数量的字符或遇到标点符号时输出
                if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"推理过程（{reasoning_count}）：{''.join(reasoning_buffer)}")
                    reasoning_buffer = []
            elif content_type == "content":
                content_count += 1
                content_buffer.append(content)
                if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"普通内容（{content_count}）：{''.join(content_buffer)}")
                    content_buffer = []
                
        # 输出剩余的内容
        if reasoning_buffer:
            logger.info(f"推理过程（最终）：{''.join(reasoning_buffer)}")
        if content_buffer:
            logger.info(f"普通内容（最终）：{''.join(content_buffer)}")
            
        # 输出统计信息
        logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个普通内容片段")
        
        if reasoning_count == 0:
            logger.warning("未收到任何推理内容！请检查以下设置:")
            logger.warning(f"1. 推理模式是否正确：{args.reasoning_mode}")
            logger.warning(f"2. API提供商 {args.provider} 是否支持推理功能")
            logger.warning(f"3. 模型 {args.model} 是否支持推理输出")
        
        logger.info("=== DeepSeek 客户端测试完成 ===")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        logger.error(f"错误类型: {type(e)}")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # 加载环境变量
    load_dotenv()
    
    asyncio.run(test_deepseek_stream(args))

if __name__ == "__main__":
    main() 