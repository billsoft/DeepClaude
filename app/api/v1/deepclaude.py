from fastapi import APIRouter, Depends, Request, Response, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import os
from pydantic import BaseModel, Field
from app.utils.logger import logger
from app.deepclaude.utils.streaming import StreamingHelper
from app.deepclaude.core import DeepClaude

router = APIRouter(
    prefix="/api/v1/deepclaude",
    tags=["deepclaude"]
)

# 用于缓存 DeepClaude 实例
_DEEPCLAUDE_INSTANCE = None

# 模型定义
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "deepclaude"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = True
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = "auto"
    save_to_db: Optional[bool] = None

def get_deepclaude():
    """获取或创建 DeepClaude 实例"""
    global _DEEPCLAUDE_INSTANCE
    if _DEEPCLAUDE_INSTANCE is None:
        logger.info("初始化 DeepClaude 实例...")
        _DEEPCLAUDE_INSTANCE = DeepClaude()
    return _DEEPCLAUDE_INSTANCE

@router.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
    raw_request: Request,
    background_tasks: BackgroundTasks,
    deepclaude: DeepClaude = Depends(get_deepclaude)
):
    """处理聊天请求，支持流式和非流式响应"""
    try:
        # 准备参数
        messages = [msg.dict() for msg in request.messages]
        model_arg = (request.temperature, request.top_p)
        
        # 处理保存到数据库的选项
        if request.save_to_db is not None:
            deepclaude.save_to_db = request.save_to_db
            
        # 处理流式请求
        if request.stream:
            logger.info(f"开始处理流式请求: model={request.model}, tools数量={len(request.tools) if request.tools else 0}")
            
            async def generate_stream():
                async for chunk in deepclaude.chat_completions_with_stream(
                    messages=messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    model=request.model
                ):
                    yield chunk
                    
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # 非流式请求
            logger.info(f"开始处理非流式请求: model={request.model}, tools数量={len(request.tools) if request.tools else 0}")
            
            response = await deepclaude.chat_completions_without_stream(
                messages=messages,
                model_arg=model_arg,
                tools=request.tools,
                tool_choice=request.tool_choice,
                model=request.model
            )
            
            return JSONResponse(content=response)
            
    except Exception as e:
        logger.error(f"处理请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reasoning")
async def get_reasoning(
    request: ChatRequest,
    deepclaude: DeepClaude = Depends(get_deepclaude)
):
    """单独获取推理结果"""
    try:
        # 准备参数
        messages = [msg.dict() for msg in request.messages]
        
        # 获取推理内容
        reasoning = await deepclaude.thinker_client.get_reasoning(
            messages=messages,
            model=request.model,
            model_arg=(request.temperature, request.top_p)
        )
        
        return JSONResponse(content={"reasoning": reasoning})
        
    except Exception as e:
        logger.error(f"获取推理时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "deepclaude"} 