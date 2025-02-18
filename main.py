import os
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

def setup_proxy():
    """设置代理配置
    根据环境变量ENABLE_PROXY的值来设置或清除系统代理
    """
    enable_proxy = os.getenv('ENABLE_PROXY', 'false').lower() == 'true'
    if enable_proxy:
        http_proxy = os.getenv('HTTP_PROXY', '')
        https_proxy = os.getenv('HTTPS_PROXY', '')
        
        if http_proxy or https_proxy:
            # 同时设置大小写版本的代理环境变量
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
            os.environ['http_proxy'] = http_proxy
            os.environ['https_proxy'] = https_proxy
            
            logger.info(f"代理已启用 - HTTP: {http_proxy}, HTTPS: {https_proxy}")
        else:
            logger.warning("代理已启用但未设置代理地址")
    else:
        # 清除所有代理设置
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(key, None)
        logger.info("代理已禁用")

def main():
    """主函数，启动FastAPI应用"""
    # 设置代理配置
    setup_proxy()
    
    # 从环境变量获取配置，如果没有则使用默认值
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 1124))
    reload = os.getenv('RELOAD', 'false').lower() == 'true'

    # 启动配置
    uvicorn.run(
        'app.main:app',
        host=host,
        port=port,
        reload=reload,
        # 在非Windows系统上启用uvloop
        loop='uvloop' if os.name != 'nt' else 'asyncio'
    )

if __name__ == '__main__':
    main()