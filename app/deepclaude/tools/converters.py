from typing import Dict, List, Any, Optional
from app.utils.logger import logger
from .validators import ToolValidator

class ToolConverter:
    """工具转换器，用于在不同格式间转换工具定义"""
    
    @staticmethod
    def openai_to_claude(tool: Dict) -> Dict:
        """将OpenAI格式工具转换为Claude格式"""
        if not ToolValidator.is_valid_openai_function(tool):
            logger.warning(f"工具格式错误，无法转换: {tool}")
            return None
            
        function_data = tool["function"]
        name = function_data.get("name", "未命名工具")
        description = function_data.get("description", "")
        
        # 处理参数
        if "parameters" in function_data and isinstance(function_data["parameters"], dict):
            parameters = function_data["parameters"].copy()
            
            # 确保参数中没有custom类型
            if "type" in parameters and parameters["type"] == "custom":
                logger.warning(f"参数中存在custom类型，正在修改为object")
                parameters["type"] = "object"
            elif "type" not in parameters:
                parameters["type"] = "object"
        else:
            parameters = {"type": "object", "properties": {}}
        
        # 创建符合Claude API要求的工具格式
        claude_tool = {
            "type": "custom",
            "name": name,
            "description": description,
            "tool_schema": parameters
        }
        
        # 确保没有custom字段
        if "custom" in claude_tool:
            logger.warning(f"转换后的工具中存在custom字段，正在移除")
            claude_tool.pop("custom", None)
        
        return claude_tool
        
    @staticmethod
    def claude_to_openai(tool: Dict) -> Dict:
        """将Claude格式工具转换为OpenAI格式"""
        if not ToolValidator.is_valid_claude_custom_tool(tool):
            logger.warning(f"工具格式错误，无法转换: {tool}")
            return None
            
        name = tool.get("name", "未命名工具")
        description = tool.get("description", "")
        schema = tool.get("tool_schema", {})
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema
            }
        }
        
        return openai_tool
        
    @staticmethod
    def fix_claude_custom_tool(tool: Dict) -> Dict:
        """修复Claude自定义工具中的嵌套type字段问题"""
        if not ToolValidator.is_valid_claude_custom_tool(tool):
            return tool
            
        if ToolValidator.has_nested_custom_type(tool):
            fixed_tool = tool.copy()
            fixed_tool["custom"] = tool["custom"].copy()
            fixed_tool["custom"].pop("type", None)
            logger.debug(f"已修复工具中的嵌套type字段: {tool.get('name', '未命名工具')}")
            return fixed_tool
            
        return tool 