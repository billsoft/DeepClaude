"""硅基流动 DeepSeek R1 API 客户端测试

这个测试脚本专门用于测试硅基流动版本的DeepSeek R1 API，
验证其推理过程（reasoning_content）和答案内容（content）的输出。

使用方法:
1. 直接运行（会自动从.env读取配置）:
   python test/test_siliconflow_deepseek.py

2. 手动指定参数（如需覆盖.env中的配置）:
   python test/test_siliconflow_deepseek.py --api-key YOUR_API_KEY [--question "您的提问"] [--debug]
"""
import os
import sys
import asyncio
import json
import argparse
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

from app.clients.deepseek_client import DeepSeekClient
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

def parse_args():
    """解析命令行参数，优先使用环境变量中的配置"""
    # 从环境变量中获取硅基流动配置
    default_api_key = os.getenv("DEEPSEEK_API_KEY")
    default_api_url = os.getenv("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
    
    parser = argparse.ArgumentParser(description='测试硅基流动 DeepSeek R1 API')
    parser.add_argument('--api-key', type=str, 
                        default=default_api_key,
                        help=f'硅基流动API密钥 (默认: {default_api_key[:8]}*** 来自环境变量)' if default_api_key else '硅基流动API密钥')
    parser.add_argument('--api-url', type=str,
                        default=default_api_url,
                        help=f'硅基流动API地址 (默认: {default_api_url})')
    parser.add_argument('--model', type=str,
                        default=default_model,
                        help=f'硅基流动模型名称 (默认: {default_model})')
    parser.add_argument('--question', type=str, 
                        default='中国大模型行业2025年将会迎来哪些机遇和挑战？',
                        help='测试问题')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    args = parser.parse_args()
    
    # 确保所有必要的参数都有值
    if not args.api_key:
        logger.error("未提供API密钥！请在.env文件中设置DEEPSEEK_API_KEY或使用--api-key参数")
        sys.exit(1)
        
    return args

async def test_siliconflow_reasoning(args):
    """测试硅基流动 DeepSeek R1 的推理功能"""
    # 记录配置信息
    logger.info("=== 硅基流动 DeepSeek-R1 API 测试开始 ===")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"API Key: {args.api_key[:8]}***")
    logger.info(f"Model: {args.model}")
    logger.info(f"测试问题: {args.question}")
    
    # 测试问题
    messages = [
        {"role": "user", "content": args.question}
    ]
    
    # 初始化客户端，指定provider为siliconflow
    client = DeepSeekClient(
        api_key=args.api_key,
        api_url=args.api_url,
        provider="siliconflow"
    )
    
    # 确保使用正确的推理模式
    os.environ["DEEPSEEK_REASONING_MODE"] = "reasoning_field"
    os.environ["IS_ORIGIN_REASONING"] = "true"
    
    try:
        logger.info("开始测试硅基流动 DeepSeek-R1 推理功能...")
        logger.debug(f"发送消息: {messages}")
        
        # 缓存推理内容和回答内容
        reasoning_buffer = []
        content_buffer = []
        
        # 统计计数器
        reasoning_count = 0
        content_count = 0
        
        # 使用get_reasoning方法获取推理过程
        async for content_type, content in client.get_reasoning(
            messages=messages,
            model=args.model
        ):
            if content_type == "reasoning":
                reasoning_count += 1
                reasoning_buffer.append(content)
                # 收集到一定数量字符或遇到标点符号时输出
                if len(''.join(reasoning_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"推理过程（{reasoning_count}）: {''.join(reasoning_buffer)}")
                    reasoning_buffer = []
            elif content_type == "content":
                content_count += 1
                content_buffer.append(content)
                # 收集到一定数量字符或遇到标点符号时输出
                if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                    logger.info(f"回答内容（{content_count}）: {''.join(content_buffer)}")
                    content_buffer = []
        
        # 输出剩余内容
        if reasoning_buffer:
            logger.info(f"推理过程（最终）: {''.join(reasoning_buffer)}")
        if content_buffer:
            logger.info(f"回答内容（最终）: {''.join(content_buffer)}")
            
        # 输出统计信息
        logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个回答内容片段")
        
        # 验证测试结果
        if reasoning_count == 0:
            logger.warning("未收到任何推理内容！请检查以下设置:")
            logger.warning("1. 确保DEEPSEEK_REASONING_MODE设置为'reasoning_field'")
            logger.warning("2. 确保IS_ORIGIN_REASONING设置为'true'")
            logger.warning("3. 确保硅基流动API支持推理输出功能")
        
        logger.info("=== 硅基流动 DeepSeek-R1 API 测试完成 ===")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        logger.error(f"错误类型: {type(e)}")

async def test_siliconflow_non_stream_api(args):
    """测试硅基流动 DeepSeek R1 的非流式API调用"""
    import aiohttp
    
    logger.info("=== 硅基流动 DeepSeek-R1 非流式API测试开始 ===")
    logger.info(f"测试问题: {args.question}")
    
    # 构建请求数据
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.question
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(args.api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API调用失败: HTTP {response.status}\n{error_text}")
                    return
                
                data = await response.json()
                
                # 记录完整响应
                logger.debug(f"API响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
                
                # 提取并显示关键信息
                if "choices" in data and len(data["choices"]) > 0:
                    # 尝试提取reasoning_content
                    reasoning = data["choices"][0].get("reasoning_content")
                    content = data["choices"][0].get("message", {}).get("content")
                    
                    if reasoning:
                        logger.info(f"推理内容: {reasoning[:200]}...")
                    else:
                        logger.warning("响应中没有推理内容")
                        
                    if content:
                        logger.info(f"回答内容: {content[:200]}...")
                    else:
                        logger.warning("响应中没有回答内容")
                        
                    # 显示token统计
                    if "usage" in data:
                        logger.info(f"Token使用情况: {data['usage']}")
                
                logger.info("=== 硅基流动 DeepSeek-R1 非流式API测试完成 ===")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)

async def test_siliconflow_stream_api_direct(args):
    """测试硅基流动 DeepSeek R1 的流式API调用（直接使用requests发送请求）"""
    import requests
    
    logger.info("=== 硅基流动 DeepSeek-R1 流式API直接调用测试开始 ===")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"API Key: {args.api_key[:8]}***")
    logger.info(f"Model: {args.model}")
    logger.info(f"测试问题: {args.question}")
    
    # 构建请求数据
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.question
            }
        ],
        "stream": True,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("发送流式API请求...")
        response = requests.post(args.api_url, json=payload, headers=headers, stream=True)
        
        if response.status_code != 200:
            logger.error(f"API调用失败: HTTP {response.status_code}\n{response.text}")
            return
        
        # 初始化缓冲区和计数器
        reasoning_buffer = []
        content_buffer = []
        reasoning_count = 0
        content_count = 0
        
        logger.info("开始接收流式响应...")
        
        # 处理流式响应
        for line in response.iter_lines():
            if not line:
                continue
                
            # 去除 "data: " 前缀并解析 JSON
            if line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8")
                
                # 处理结束标记
                if data_str == "[DONE]":
                    logger.info("收到流式响应结束标记")
                    break
                    
                try:
                    data = json.loads(data_str)
                    logger.debug(f"收到数据: {json.dumps(data, ensure_ascii=False)}")
                    
                    # 提取推理内容
                    if "choices" in data and len(data["choices"]) > 0:
                        # 尝试提取reasoning_content
                        reasoning = data["choices"][0].get("reasoning_content")
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        # 处理推理内容
                        if reasoning:
                            reasoning_count += 1
                            reasoning_buffer.append(reasoning)
                            # 收集到一定数量字符或遇到标点符号时输出
                            if len(''.join(reasoning_buffer)) >= 50 or any(p in reasoning for p in '。，！？.!?'):
                                logger.info(f"推理过程（{reasoning_count}）: {''.join(reasoning_buffer)}")
                                reasoning_buffer = []
                        
                        # 处理回答内容
                        if content:
                            content_count += 1
                            content_buffer.append(content)
                            # 收集到一定数量字符或遇到标点符号时输出
                            if len(''.join(content_buffer)) >= 50 or any(p in content for p in '。，！？.!?'):
                                logger.info(f"回答内容（{content_count}）: {''.join(content_buffer)}")
                                content_buffer = []
                
                except json.JSONDecodeError:
                    logger.error(f"无法解析 JSON: {data_str}")
        
        # 输出剩余内容
        if reasoning_buffer:
            logger.info(f"推理过程（最终）: {''.join(reasoning_buffer)}")
        if content_buffer:
            logger.info(f"回答内容（最终）: {''.join(content_buffer)}")
            
        # 输出统计信息
        logger.info(f"测试完成 - 收到 {reasoning_count} 个推理片段，{content_count} 个回答内容片段")
        logger.info("=== 硅基流动 DeepSeek-R1 流式API直接调用测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)
        logger.error(f"错误类型: {type(e)}")

def check_environment():
    """检查环境变量是否配置正确"""
    # 检查是否已配置硅基流动
    provider = os.getenv('REASONING_PROVIDER', '').lower()
    if provider == 'siliconflow':
        logger.info("检测到环境变量REASONING_PROVIDER=siliconflow")
    else:
        logger.warning(f"当前REASONING_PROVIDER={provider}，非siliconflow，但仍会继续测试")
    
    # 检查API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        logger.info(f"已从环境变量中读取API密钥: {api_key[:8]}***")
    else:
        logger.warning("环境变量中未设置DEEPSEEK_API_KEY")
    
    # 检查API URL
    api_url = os.getenv('DEEPSEEK_API_URL')
    if api_url:
        logger.info(f"已从环境变量中读取API URL: {api_url}")
    else:
        logger.warning("环境变量中未设置DEEPSEEK_API_URL，将使用默认值")
    
    # 检查推理模式
    is_origin_reasoning = os.getenv('IS_ORIGIN_REASONING', '').lower() == 'true'
    reasoning_mode = os.getenv('DEEPSEEK_REASONING_MODE', '')
    if is_origin_reasoning and reasoning_mode == 'reasoning_field':
        logger.info("推理模式配置正确")
    else:
        logger.warning(f"当前推理模式可能不适合硅基流动API：IS_ORIGIN_REASONING={is_origin_reasoning}, DEEPSEEK_REASONING_MODE={reasoning_mode}")
        logger.warning("已自动设置为正确的推理模式")

def main():
    """主函数"""
    # 优先查看环境变量配置
    check_environment()
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
    else:
        os.environ['LOG_LEVEL'] = 'INFO'
    
    # 创建事件循环并运行测试
    loop = asyncio.get_event_loop()
    
    # 运行测试
    loop.run_until_complete(test_siliconflow_reasoning(args))
    loop.run_until_complete(test_siliconflow_non_stream_api(args))
    loop.run_until_complete(test_siliconflow_stream_api_direct(args))
    
    # 关闭事件循环
    loop.close()

if __name__ == "__main__":
    main() 