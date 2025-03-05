"""流式响应处理工具"""

import json
from typing import Dict, Any

class StreamingHelper:
    """流式响应辅助工具"""
    
    @staticmethod
    def format_chunk_response(content: str, role: str = "assistant", chat_id: str = None, 
                             created_time: int = None, model: str = "deepclaude", 
                             is_reasoning: bool = False, finish_reason: str = None) -> str:
        """格式化流式响应块"""
        response = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": role,
                    "content": content
                },
                "finish_reason": finish_reason
            }]
        }
        
        if is_reasoning:
            response["is_reasoning"] = True
            response["choices"][0]["delta"]["reasoning"] = True
            
        return f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def format_done_marker() -> str:
        """生成流式响应结束标记"""
        return "data: [DONE]\n\n" 