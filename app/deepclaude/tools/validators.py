from typing import Dict, List, Any, Optional
from app.utils.logger import logger

class ToolValidator:
    """工具验证器，用于验证工具格式的有效性"""
    
    @staticmethod
    def is_valid_openai_function(tool: Dict) -> bool:
        """验证是否为有效的OpenAI函数工具格式"""
        if not isinstance(tool, dict):
            return False
            
        if "function" not in tool:
            return False
            
        function = tool["function"]
        if not isinstance(function, dict):
            return False
            
        if "name" not in function:
            return False
            
        return True
        
    @staticmethod
    def is_valid_claude_custom_tool(tool: Dict) -> bool:
        """验证是否为有效的Claude自定义工具格式"""
        if not isinstance(tool, dict):
            return False
            
        if "type" not in tool or tool["type"] != "custom":
            return False
            
        if "name" not in tool:
            return False
            
        return True
        
    @staticmethod
    def has_nested_custom_type(tool: Dict) -> bool:
        """检查Claude自定义工具中是否有嵌套的type字段"""
        if not isinstance(tool, dict):
            return False
            
        # 检查顶层是否有custom字段
        if "custom" in tool and isinstance(tool["custom"], dict) and "type" in tool["custom"]:
            return True
            
        # 检查tool_schema中是否有type=custom
        if "type" in tool and tool["type"] == "custom" and "tool_schema" in tool:
            tool_schema = tool["tool_schema"]
            if isinstance(tool_schema, dict) and "type" in tool_schema and tool_schema["type"] == "custom":
                return True
                
        return False
        
    @staticmethod
    def validate_claude_tool(tool: Dict) -> tuple[bool, list[str]]:
        """验证工具格式是否符合Claude API要求
        
        返回:
            (是否有效, 错误消息列表)
        """
        errors = []
        if not isinstance(tool, dict):
            return False, ["工具必须是字典类型"]
            
        # 检查必要字段
        required_fields = ["name", "description", "input_schema"]
        for field in required_fields:
            if field not in tool:
                errors.append(f"缺少必要字段: {field}")
        
        # 检查input_schema字段
        if "input_schema" in tool:
            input_schema = tool["input_schema"]
            if not isinstance(input_schema, dict):
                errors.append("input_schema字段必须是字典类型")
            else:
                # 检查input_schema的结构
                if "type" not in input_schema:
                    errors.append("input_schema必须包含type字段")
                elif input_schema["type"] != "object":
                    errors.append("input_schema的type字段值必须为'object'")
                if "properties" not in input_schema:
                    errors.append("input_schema必须包含properties字段")
                elif not isinstance(input_schema["properties"], dict):
                    errors.append("properties字段必须是字典类型")
            
        return len(errors) == 0, errors 