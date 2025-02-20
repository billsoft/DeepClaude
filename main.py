import os
import uvicorn
from dotenv import load_dotenv
from app.utils.logger import logger

# 加载环境变量
load_dotenv()

def main():
    """主函数，启动FastAPI应用"""
    # 从环境变量获取配置，如果没有则使用默认值
    host = os.getenv('HOST', '::')
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